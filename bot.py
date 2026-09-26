"""
magicpin AI Challenge — Vera Merchant AI Assistant Bot Server
=============================================================
FastAPI server exposing:
  - GET  /v1/healthz
  - GET  /v1/metadata
  - POST /v1/context
  - POST /v1/tick
  - POST /v1/reply
  - POST /v1/reset
  - POST /v1/teardown

Features:
  - In-memory state store enforcing idempotency by (scope, context_id, version).
  - Gemini LLM integration (gemini-2.5-flash / gemini-2.0-flash) at temperature=0.
  - Instant deterministic fallback composer on timeout or 429 rate limits.
  - Zero hardcoded test pairs — 100% dynamic 4-context composition.
  - Full environment variable configuration support.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field
import uvicorn

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vera_bot")

# -----------------------------------------------------------------------------
# Configuration & Environment Variables
# -----------------------------------------------------------------------------
BOT_URL = os.environ.get("BOT_URL", "http://localhost:8000")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "") or GEMINI_API_KEY
if not GEMINI_API_KEY and LLM_API_KEY:
    GEMINI_API_KEY = LLM_API_KEY

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Server Start Timestamp
START_TIME = time.time()

# Valid Context Scopes
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# -----------------------------------------------------------------------------
# In-Memory State Store with Idempotency
# -----------------------------------------------------------------------------

class ContextStore:
    """
    Thread-safe in-memory context store enforcing idempotency by (scope, context_id, version).
    Atomically replaces lower versions and ignores/rejects duplicate or stale versions with 409.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Key: (scope, context_id) -> {"version": int, "payload": dict, "delivered_at": str, "stored_at": str}
        self._contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}

    async def put_context(
        self, scope: str, context_id: str, version: int, payload: Dict[str, Any], delivered_at: str
    ) -> Tuple[bool, str, Optional[int]]:
        if scope not in VALID_SCOPES:
            return False, "invalid_scope", None

        async with self._lock:
            key = (scope, context_id)
            current = self._contexts.get(key)
            if current is not None:
                cur_ver = current["version"]
                # Duplicate or lower versions: enforce idempotency and ignore
                if version <= cur_ver:
                    return False, "stale_version", cur_ver

            # Incoming version is higher or key is new: atomically insert/replace
            stored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self._contexts[key] = {
                "version": version,
                "payload": payload,
                "delivered_at": delivered_at,
                "stored_at": stored_at,
            }
            return True, f"ack_{context_id}_v{version}", None

    async def get_context(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            entry = self._contexts.get((scope, context_id))
            return entry.get("payload") if entry else None

    async def get_counts(self) -> Dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        async with self._lock:
            for (scope, _), _ in self._contexts.items():
                if scope in counts:
                    counts[scope] += 1
        return counts

    async def reset(self) -> None:
        async with self._lock:
            self._contexts.clear()


class ConversationStore:
    """Tracks active conversations, suppression keys, auto-reply counts, and opt-outs."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.suppressed_keys: Set[str] = set()
        self.auto_reply_counts: Dict[str, int] = {}
        self.opted_out_merchants: Set[str] = set()
        self.closed_conversations: Set[str] = set()

    async def is_suppressed(self, suppression_key: str) -> bool:
        async with self._lock:
            return suppression_key in self.suppressed_keys

    async def mark_suppressed(self, suppression_key: str) -> None:
        async with self._lock:
            if suppression_key:
                self.suppressed_keys.add(suppression_key)

    async def is_opted_out(self, merchant_id: str) -> bool:
        async with self._lock:
            return merchant_id in self.opted_out_merchants

    async def mark_opted_out(self, merchant_id: str) -> None:
        async with self._lock:
            self.opted_out_merchants.add(merchant_id)

    async def is_closed(self, conv_id: str) -> bool:
        async with self._lock:
            return conv_id in self.closed_conversations

    async def mark_closed(self, conv_id: str) -> None:
        async with self._lock:
            self.closed_conversations.add(conv_id)

    async def record_turn(self, conv_id: str, from_role: str, message: str) -> None:
        async with self._lock:
            self.conversations.setdefault(conv_id, []).append({
                "from_role": from_role,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            })

    async def increment_auto_reply(self, conv_id: str) -> int:
        async with self._lock:
            cnt = self.auto_reply_counts.get(conv_id, 0) + 1
            self.auto_reply_counts[conv_id] = cnt
            return cnt

    async def reset(self) -> None:
        async with self._lock:
            self.conversations.clear()
            self.suppressed_keys.clear()
            self.auto_reply_counts.clear()
            self.opted_out_merchants.clear()
            self.closed_conversations.clear()


# Initialize state stores
context_store = ContextStore()
conversation_store = ConversationStore()

# -----------------------------------------------------------------------------
# Deterministic Fallback Composer (Instant, Reliable, High-Specificity)
# -----------------------------------------------------------------------------

def compose_deterministic(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic 4-context composer. Serves as instant fallback on LLM
    timeout, 429 rate limit, or when no API key is configured.
    """
    trigger_kind = trigger.get("kind", "general_update")
    trigger_payload = trigger.get("payload", {})
    trigger_id = trigger.get("id", "trg_unknown")
    suppression_key = trigger.get("suppression_key", f"suppress:{trigger_id}")

    merchant_id = merchant.get("merchant_id", "m_unknown")
    identity = merchant.get("identity", {})
    merchant_name = identity.get("name", "Partner")
    owner_first_name = identity.get("owner_first_name", "")
    locality = identity.get("locality", "")
    city = identity.get("city", "")

    cat_slug = category.get("slug", "")
    peer_stats = category.get("peer_stats", {})
    digest_items = category.get("digest", [])

    performance = merchant.get("performance", {})
    views = performance.get("views", 0)
    calls = performance.get("calls", 0)

    # Determine Owner Salutation matching Category Voice
    if cat_slug == "dentists":
        salutation = f"Dr. {owner_first_name}" if owner_first_name else (
            f"Dr. {merchant_name.replace('Dr. ', '')}" if "Dr." in merchant_name else "Doctor"
        )
    elif owner_first_name:
        salutation = owner_first_name
    else:
        salutation = merchant_name

    # Check for Active Offers
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    primary_offer_title = offers[0].get("title", "") if offers else ""
    if not primary_offer_title:
        cat_offers = category.get("offer_catalog", [])
        if cat_offers:
            primary_offer_title = cat_offers[0].get("title", "")

    # Customer-Scoped Trigger: Recall or Appointment Followup
    if customer is not None:
        cx_identity = customer.get("identity", {})
        cx_name = cx_identity.get("name", "Valued Customer")
        cx_lang = cx_identity.get("language_preference", "en").lower()

        slots = trigger_payload.get("available_slots", [])
        slot_text = ""
        if slots:
            slot_labels = [s.get("label", "") for s in slots if s.get("label")]
            if len(slot_labels) >= 2:
                slot_text = f"{slot_labels[0]} or {slot_labels[1]}"
            elif slot_labels:
                slot_text = slot_labels[0]

        service_due = trigger_payload.get("service_due", "routine checkup").replace("_", " ")
        offer_mention = f"{primary_offer_title}" if primary_offer_title else "routine service"

        if "hi" in cx_lang:
            if slot_text:
                body = (
                    f"Hi {cx_name}, {merchant_name} here. Aapka {service_due} recall due hai. "
                    f"Aapke liye 2 slots ready hain: {slot_text}. {offer_mention}. "
                    f"Reply 1 or 2 to confirm, ya apna preferred time batayein."
                )
            else:
                body = (
                    f"Hi {cx_name}, {merchant_name} here. Aapka {service_due} recall due hai. "
                    f"Special offer available: {offer_mention}. "
                    f"Reply karein to book your slot this week."
                )
        else:
            if slot_text:
                body = (
                    f"Hi {cx_name}, {merchant_name} here. Your {service_due} is due. "
                    f"We have reserved priority slots for you: {slot_text}. {offer_mention}. "
                    f"Reply 1 or 2 to confirm, or tell us a time that works best for you."
                )
            else:
                body = (
                    f"Hi {cx_name}, {merchant_name} here. Your {service_due} is due. "
                    f"Special offer: {offer_mention}. Reply to schedule your visit this week."
                )

        conv_id = f"conv_{merchant_id}_{cx_identity.get('customer_id', 'cx')}_{uuid.uuid4().hex[:6]}"
        return {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer.get("customer_id"),
            "send_as": "merchant_on_behalf",
            "trigger_id": trigger_id,
            "template_name": f"{cat_slug}_customer_recall_v1",
            "template_params": [cx_name, merchant_name, service_due, slot_text or "priority slots", offer_mention],
            "body": body,
            "cta": "multi_choice_slot" if slot_text else "open_ended",
            "suppression_key": suppression_key,
            "rationale": f"Customer-scoped {trigger_kind} sent on behalf of merchant. Honors customer preferences with verifiable slots.",
        }

    # Merchant-Scoped Trigger Handling
    body = ""
    cta = "open_ended"
    template_params = []
    rationale = ""

    if trigger_kind == "research_digest":
        top_item_id = trigger_payload.get("top_item_id")
        digest_item = next((d for d in digest_items if d.get("id") == top_item_id), None)
        if not digest_item and digest_items:
            digest_item = digest_items[0]

        title = digest_item.get("title", "new clinical research findings") if digest_item else "recent clinical research"
        source = digest_item.get("source", "industry journal") if digest_item else "Journal findings"
        summary = digest_item.get("summary", "") if digest_item else ""

        body = (
            f"{salutation}, {source} just published an update relevant to {locality}: '{title}'. "
            f"{summary} Would you like me to pull the 2-minute summary and draft a patient education WhatsApp post? — {source}"
        )
        cta = "open_ended"
        template_params = [salutation, title, source]
        rationale = f"Research digest notification with source citation ({source}) and clinical relevance for {cat_slug}."

    elif trigger_kind in ("regulation_change", "compliance"):
        top_item_id = trigger_payload.get("top_item_id")
        digest_item = next((d for d in digest_items if d.get("id") == top_item_id), None)
        deadline = trigger_payload.get("deadline_iso", "upcoming deadline")
        title = digest_item.get("title", "regulatory compliance guidelines") if digest_item else "new regulatory norms"
        source = digest_item.get("source", "official circular") if digest_item else "regulatory notice"

        body = (
            f"{salutation}, new compliance notice from {source}: '{title}'. Effective deadline is {deadline}. "
            f"Would you like me to share a quick 3-point checklist to keep {merchant_name} compliant?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, title, deadline]
        rationale = f"Regulatory compliance alert for {merchant_name} with verified deadline ({deadline})."

    elif trigger_kind in ("perf_dip", "seasonal_perf_dip"):
        metric = trigger_payload.get("metric", "views")
        delta_pct = trigger_payload.get("delta_pct", -0.2)
        drop_pct_str = f"{abs(delta_pct) * 100:.0f}%"
        curr_val = views if metric == "views" else calls
        peer_avg = peer_stats.get("avg_ctr", 0.03)

        body = (
            f"{salutation}, we noticed your Google Profile {metric} dropped {drop_pct_str} this week ({curr_val} total). "
            f"Local peer CTR median in {city} is {peer_avg * 100:.1f}%. "
            f"Would you like me to launch a fresh post for '{primary_offer_title or 'your core service'}' to boost walk-ins?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, metric, drop_pct_str, primary_offer_title]
        rationale = f"Performance alert citing exact drop ({drop_pct_str}) and peer benchmark to drive proactive offer refresh."

    elif trigger_kind == "perf_spike":
        metric = trigger_payload.get("metric", "views")
        delta_pct = trigger_payload.get("delta_pct", 0.25)
        spike_pct_str = f"{abs(delta_pct) * 100:.0f}%"
        curr_val = views if metric == "views" else calls

        body = (
            f"{salutation}, great momentum — your Google profile {metric} surged {spike_pct_str} this week ({curr_val} total)! "
            f"Shall we publish an update featuring '{primary_offer_title or 'your top package'}' to convert these searches into bookings?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, metric, spike_pct_str]
        rationale = f"Capitalizing on verified traffic spike ({spike_pct_str}) with low-friction post scheduling."

    elif trigger_kind == "renewal_due":
        days_left = trigger_payload.get("days_remaining", 7)
        plan = trigger_payload.get("plan", "Growth")
        amount = trigger_payload.get("renewal_amount", 4999)

        body = (
            f"{salutation}, your {plan} plan for {merchant_name} has {days_left} days remaining. "
            f"Renewal is ₹{amount:,}. Would you like me to generate your instant renewal link to keep campaigns running smoothly?"
        )
        cta = "binary_confirm_cancel"
        template_params = [salutation, plan, str(days_left), f"₹{amount:,}"]
        rationale = f"Subscription renewal reminder citing specific plan ({plan}) and days remaining ({days_left})."

    elif trigger_kind == "festival_upcoming":
        festival = trigger_payload.get("festival", "upcoming celebration")
        date = trigger_payload.get("date", "this month")
        days_until = trigger_payload.get("days_until", 7)

        body = (
            f"{salutation}, {festival} is {days_until} days away ({date}). "
            f"Local searches for {cat_slug} in {locality} peak significantly before festivals. "
            f"Shall we schedule a '{primary_offer_title or 'festive bundle'}' special post for tomorrow?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, festival, str(days_until), primary_offer_title]
        rationale = f"Seasonal festival campaign timed {days_until} days before {festival}."

    elif trigger_kind == "milestone_reached":
        metric = trigger_payload.get("metric", "reviews").replace("_", " ")
        curr_val = trigger_payload.get("value_now", 98)
        milestone = trigger_payload.get("milestone_value", 100)
        gap = max(1, milestone - curr_val)

        body = (
            f"{salutation}, {merchant_name} is at {curr_val} {metric} — only {gap} more to cross the {milestone} milestone! "
            f"Would you like me to prepare a quick review-request message to send to your happy clients today?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, str(curr_val), str(milestone), str(gap)]
        rationale = f"Milestone celebration and nudge to reach {milestone} {metric}."

    elif trigger_kind == "review_theme_emerged":
        theme = trigger_payload.get("theme", "service feedback").replace("_", " ")
        occurrences = trigger_payload.get("occurrences_30d", 3)
        common_quote = trigger_payload.get("common_quote", "")

        quote_clause = f" (e.g. \"{common_quote}\")" if common_quote else ""
        body = (
            f"{salutation}, we detected {occurrences} customer mentions of '{theme}' in recent feedback{quote_clause}. "
            f"Would you like me to draft an operational response and a customer reassurance post for {merchant_name}?"
        )
        cta = "open_ended"
        template_params = [salutation, theme, str(occurrences)]
        rationale = f"Review intelligence trigger addressing '{theme}' with concrete occurrence count ({occurrences})."

    elif trigger_kind == "winback_eligible":
        days_expiry = trigger_payload.get("days_since_expiry", 30)
        lapsed_count = trigger_payload.get("lapsed_customers_added_since_expiry", 15)

        body = (
            f"{salutation}, {merchant_name} has {lapsed_count} customers due for recall since your plan paused {days_expiry} days ago. "
            f"Would you like to reactivate your campaign today to reconnect with them?"
        )
        cta = "binary_yes_no"
        template_params = [salutation, str(lapsed_count), str(days_expiry)]
        rationale = f"Winback nudge highlighting {lapsed_count} lapsed customers and {days_expiry} days dormancy."

    elif trigger_kind == "curious_ask_due":
        body = (
            f"{salutation}, quick question from our {cat_slug} pulse check in {locality}: "
            f"which service or appointment type has had the highest inquiry volume this week? "
            f"Reply with the service name and I'll optimize your profile visibility for it."
        )
        cta = "open_ended"
        template_params = [salutation, cat_slug, locality]
        rationale = f"Curiosity-driven prompt for {cat_slug} merchant to encourage organic dialogue."

    else:
        body = (
            f"{salutation}, regarding your {cat_slug} business at {locality}: "
            f"we have an opportunity to optimize your profile around '{primary_offer_title or 'your core offering'}'. "
            f"Would you like me to draft a quick post for your Google Business Profile today?"
        )
        cta = "open_ended"
        template_params = [salutation, cat_slug, primary_offer_title]
        rationale = f"Dynamic composition for trigger kind '{trigger_kind}' referencing verified merchant details."

    conv_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}"
    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": None,
        "send_as": "vera",
        "trigger_id": trigger_id,
        "template_name": f"{cat_slug}_{trigger_kind}_v1",
        "template_params": template_params,
        "body": body,
        "cta": cta,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }


# -----------------------------------------------------------------------------
# LLM Integration: Gemini with Temperature=0 & Instant Fallback
# -----------------------------------------------------------------------------

async def call_gemini_composer(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Calls Google Gemini REST API (gemini-2.5-flash or gemini-2.0-flash) with temperature=0.
    Returns parsed JSON action or None on failure/429/timeout.
    """
    prompt = (
        "You are Vera, magicpin's WhatsApp Merchant AI Assistant in India.\n"
        "Compose the next proactive WhatsApp message adhering to the 4-context framework.\n\n"
        f"=== CATEGORY CONTEXT ===\n{json.dumps(category, indent=2)}\n\n"
        f"=== MERCHANT CONTEXT ===\n{json.dumps(merchant, indent=2)}\n\n"
        f"=== TRIGGER CONTEXT ===\n{json.dumps(trigger, indent=2)}\n\n"
        f"=== CUSTOMER CONTEXT ===\n{json.dumps(customer, indent=2) if customer else 'None (merchant-facing)'}\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. SPECIFICITY: Include verifiable numbers, percentages, prices (₹), dates, and source citations (e.g. '— JIDA Oct 2026 p.14').\n"
        "2. CATEGORY VOICE: Clinical/peer for dentists (use 'Dr.' prefix); warm for salons; operator for restaurants; coaching for gyms; trustworthy for pharmacies. Never use taboo words.\n"
        "3. MERCHANT FIT: Personalize to owner name, business name, locality, and active offers.\n"
        "4. TRIGGER RELEVANCE: Focus directly on the trigger event.\n"
        "5. ENGAGEMENT: Compelling CTA with low friction.\n"
        "6. NO URLs: Meta WhatsApp policy forbids raw URLs in outbound templates.\n"
        "7. If customer is present: send_as must be 'merchant_on_behalf' with customer name and appointment slots.\n"
        "8. Output MUST be valid JSON with this exact schema:\n"
        "{\n"
        '  "conversation_id": "string",\n'
        '  "merchant_id": "string",\n'
        '  "customer_id": null or "string",\n'
        '  "send_as": "vera" or "merchant_on_behalf",\n'
        '  "trigger_id": "string",\n'
        '  "template_name": "string",\n'
        '  "template_params": ["string"],\n'
        '  "body": "string",\n'
        '  "cta": "open_ended" or "binary_yes_no" or "multi_choice_slot" or "binary_confirm_cancel",\n'
        '  "suppression_key": "string",\n'
        '  "rationale": "string"\n'
        "}"
    )

    models_to_try = [GEMINI_MODEL]
    if "2.5" in GEMINI_MODEL:
        models_to_try.append("gemini-2.0-flash")
    elif "2.0" in GEMINI_MODEL:
        models_to_try.append("gemini-2.5-flash")
    else:
        models_to_try.extend(["gemini-2.5-flash", "gemini-2.0-flash"])

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
            },
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 429:
                    logger.warning(f"Gemini {model_name} rate limit (429) hit. Falling back immediately.")
                    return None
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return None
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "body" in parsed:
                    return parsed
        except (httpx.TimeoutException, httpx.RequestError) as ex:
            logger.warning(f"Gemini {model_name} network/timeout: {ex}. Proceeding to fallback.")
            return None
        except Exception as ex:
            logger.warning(f"Gemini {model_name} unexpected error: {ex}. Proceeding to fallback.")
            return None

    return None


async def compose(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main compose function. Calls Gemini at temperature=0 if API key is present.
    Instant fallback to deterministic composer on timeout, 429, or errors.
    """
    api_key = GEMINI_API_KEY or LLM_API_KEY
    if api_key and LLM_PROVIDER.lower() == "gemini":
        try:
            gemini_result = await call_gemini_composer(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
                api_key=api_key,
            )
            if gemini_result and isinstance(gemini_result, dict) and "body" in gemini_result:
                # Ensure all required fields are present
                merchant_id = merchant.get("merchant_id", "m_unknown")
                trigger_id = trigger.get("id", "trg_unknown")
                cat_slug = category.get("slug", "category")
                trigger_kind = trigger.get("kind", "update")

                gemini_result.setdefault("conversation_id", f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}")
                gemini_result.setdefault("merchant_id", merchant_id)
                gemini_result.setdefault("customer_id", customer.get("customer_id") if customer else None)
                gemini_result.setdefault("send_as", "merchant_on_behalf" if customer else "vera")
                gemini_result.setdefault("trigger_id", trigger_id)
                gemini_result.setdefault("template_name", f"{cat_slug}_{trigger_kind}_v1")
                gemini_result.setdefault("template_params", [])
                gemini_result.setdefault("cta", "open_ended")
                gemini_result.setdefault("suppression_key", trigger.get("suppression_key", f"suppress:{trigger_id}"))
                gemini_result.setdefault("rationale", "Composed via Gemini at temperature=0 adhering to 4 contexts.")
                return gemini_result
        except Exception as e:
            logger.warning(f"Gemini composer exception: {e}. Executing instant fallback.")

    # Instant deterministic fallback
    return compose_deterministic(category=category, merchant=merchant, trigger=trigger, customer=customer)


# -----------------------------------------------------------------------------
# FastAPI Application & Endpoints
# -----------------------------------------------------------------------------

app = FastAPI(
    title="magicpin Vera AI Merchant Assistant",
    description="WhatsApp Merchant AI Assistant compliant with magicpin AI Challenge specifications.",
    version="1.0.0",
)


# Request & Response Models

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class TickRequest(BaseModel):
    now: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    available_triggers: List[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    turn_number: int = 1


# -----------------------------------------------------------------------------
# 1. GET /v1/healthz
# -----------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    uptime = int(time.time() - START_TIME)
    counts = await context_store.get_counts()
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "contexts_loaded": counts,
    }


# -----------------------------------------------------------------------------
# 2. GET /v1/metadata
# -----------------------------------------------------------------------------
@app.get("/v1/metadata")
async def metadata():
    active_model = f"{LLM_PROVIDER} ({GEMINI_MODEL})" if (GEMINI_API_KEY or LLM_API_KEY) else "deterministic-fallback-composer"
    return {
        "team_name": "Team Vera",
        "team_members": ["AI Engineer"],
        "model": active_model,
        "approach": "Gemini temperature=0 composition with deterministic fallback + 4-context framework",
        "contact_email": "candidate@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-04-26T08:00:00Z",
    }


# -----------------------------------------------------------------------------
# 3. POST /v1/context (In-Memory State Store with Idempotency)
# -----------------------------------------------------------------------------
@app.post("/v1/context")
async def push_context(body: ContextRequest):
    accepted, result_or_reason, current_version = await context_store.put_context(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
        delivered_at=body.delivered_at,
    )

    if not accepted:
        if result_or_reason == "invalid_scope":
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "accepted": False,
                    "reason": "invalid_scope",
                    "details": f"Scope '{body.scope}' is invalid. Allowed scopes: {sorted(list(VALID_SCOPES))}",
                },
            )
        # Duplicate or lower version -> Idempotency conflict / stale version (409)
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )

    return {
        "accepted": True,
        "ack_id": result_or_reason,
        "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# -----------------------------------------------------------------------------
# 4. POST /v1/tick (Proactive Trigger Handling — No Hardcoded Pairs)
# -----------------------------------------------------------------------------
@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions = []

    for trigger_id in body.available_triggers:
        # 1. Fetch TriggerContext
        trigger = await context_store.get_context("trigger", trigger_id)
        if not trigger:
            continue

        # Check suppression
        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and await conversation_store.is_suppressed(suppression_key):
            continue

        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        # Check merchant opt-out
        if await conversation_store.is_opted_out(merchant_id):
            continue

        # 2. Fetch MerchantContext
        merchant = await context_store.get_context("merchant", merchant_id)
        if not merchant:
            continue

        # 3. Fetch CategoryContext
        cat_slug = merchant.get("category_slug", "")
        category = await context_store.get_context("category", cat_slug)
        if not category:
            cat_slug_fallback = trigger.get("payload", {}).get("category", "")
            category = await context_store.get_context("category", cat_slug_fallback)
            if not category:
                category = {"slug": cat_slug or "general"}

        # 4. Fetch CustomerContext if customer scope
        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = await context_store.get_context("customer", customer_id)

        # Compose action via Gemini with deterministic fallback
        action = await compose(category=category, merchant=merchant, trigger=trigger, customer=customer)
        actions.append(action)

        # Mark suppression key active
        if suppression_key:
            await conversation_store.mark_suppressed(suppression_key)

        # Enforce rate cap per tick
        if len(actions) >= 20:
            break

    return {"actions": actions}


# -----------------------------------------------------------------------------
# 5. POST /v1/reply (Conversational Reply Engine)
# -----------------------------------------------------------------------------
@app.post("/v1/reply")
async def handle_reply(body: ReplyRequest):
    conv_id = body.conversation_id
    raw_msg = (body.message or "").strip()
    msg_lower = raw_msg.lower()

    # Record turn
    await conversation_store.record_turn(conv_id, body.from_role, raw_msg)

    # If conversation is already closed
    if await conversation_store.is_closed(conv_id):
        return {
            "action": "end",
            "rationale": "Conversation previously concluded.",
        }

    # 1. Hostile / Explicit Opt-Out Detection
    hostile_patterns = [
        r"\bstop\b", r"\bunsubscribe\b", r"\bspam\b", r"\buseless\b",
        r"\bnot interested\b", r"\bdon'?t message\b", r"\bstop messaging\b",
        r"\bleave me alone\b", r"\bbothering me\b", r"\bquit\b"
    ]
    if any(re.search(pat, msg_lower) for pat in hostile_patterns):
        if body.merchant_id:
            await conversation_store.mark_opted_out(body.merchant_id)
        await conversation_store.mark_closed(conv_id)
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out or expressed frustration. Suppressed and closed conversation immediately.",
        }

    # 2. WhatsApp Business Auto-Reply Detection
    auto_reply_patterns = [
        "thank you for contacting",
        "thanks for contacting",
        "respond shortly",
        "will get back to you",
        "team will respond",
        "currently unavailable",
        "automated response",
        "auto-reply",
        "busy right now",
    ]
    is_auto_reply = any(pat in msg_lower for pat in auto_reply_patterns)
    if is_auto_reply:
        count = await conversation_store.increment_auto_reply(conv_id)
        if body.turn_number >= 3 or count >= 2:
            await conversation_store.mark_closed(conv_id)
            return {
                "action": "end",
                "rationale": "Canned auto-reply detected repeatedly with no owner response. Ending conversation to prevent looping.",
            }
        else:
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Detected canned WhatsApp Business auto-reply. Backing off 4 hours to wait for business owner.",
            }

    # 3. Intent Transition / Commitment Detection
    commitment_patterns = [
        "ok lets do it", "ok, let's do it", "let's do it", "lets do it",
        "whats next", "what's next", "confirm", "proceed", "go ahead",
        "yes please", "yes send", "send the abstract", "draft it",
        "send it", "do it", "sure send", "yes draft"
    ]
    if any(pat in msg_lower for pat in commitment_patterns):
        # Must switch immediately to ACTION mode (no qualifying questions)
        return {
            "action": "send",
            "body": "Done! Sending the abstract now. Here is your draft post ready to review: 'Special update from our clinic — drop us a note for priority booking.' Reply CONFIRM to proceed with scheduling tomorrow 10am.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant explicitly committed. Switched from qualification to immediate drafting and action execution with binary confirmation.",
        }

    # 4. Off-Topic / Out of Scope (e.g. GST, Taxes, Legal)
    out_of_scope_patterns = [r"\bgst\b", r"\btax\b", r"\bincome tax\b", r"\baccounting\b", r"\baudit\b"]
    if any(re.search(pat, msg_lower) for pat in out_of_scope_patterns):
        return {
            "action": "send",
            "body": "I'll have to leave GST and tax filing to your CA, as that's outside what Vera handles. Coming back to our marketing plan — shall we proceed with the post draft we discussed?",
            "cta": "binary_yes_no",
            "rationale": "Politely declined out-of-scope query and steered back to the core marketing objective.",
        }

    # 5. General Engagement / Follow-on Continuation
    return {
        "action": "send",
        "body": "Understood! I will prepare the draft and details for your Google Business Profile right away. Shall I schedule it for tomorrow at 10am?",
        "cta": "binary_yes_no",
        "rationale": "Acknowledged merchant input and advanced the conversation towards action.",
    }


# -----------------------------------------------------------------------------
# Reset & Teardown Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/reset")
async def reset_state():
    await context_store.reset()
    await conversation_store.reset()
    return {"status": "ok", "message": "State reset successfully."}


@app.post("/v1/teardown")
async def teardown():
    await context_store.reset()
    await conversation_store.reset()
    return {"status": "ok", "message": "State wiped."}


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, log_level="info")
