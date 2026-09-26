"""
Comprehensive test suite for bot.py FastAPI server.
Tests:
- GET /v1/healthz
- GET /v1/metadata
- POST /v1/context (idempotency, duplicate ignore, higher version replacement, invalid scope)
- POST /v1/tick (dynamic 4-context composition, no hardcoded test pairs, suppression)
- POST /v1/reply (auto-reply detection, intent transition, hostility handling, off-topic)
"""

import asyncio
from fastapi.testclient import TestClient
import pytest
from bot import app, context_store, conversation_store


client = TestClient(app)


def setup_function():
    """Reset state before each test."""
    asyncio.run(context_store.reset())
    asyncio.run(conversation_store.reset())


def test_healthz_initial():
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert data["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


def test_metadata():
    response = client.get("/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "team_name" in data
    assert "model" in data
    assert "approach" in data
    assert "version" in data
    assert "submitted_at" in data


def test_context_push_and_idempotency():
    # 1. Push category version 1
    cat_payload = {
        "slug": "dentists",
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
        "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ ₹299"}],
        "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030},
    }
    resp1 = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": cat_payload,
    })
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["accepted"] is True
    assert data1["ack_id"] == "ack_dentists_v1"
    assert "stored_at" in data1

    # Verify counts in healthz
    healthz_resp = client.get("/v1/healthz")
    assert healthz_resp.json()["contexts_loaded"]["category"] == 1

    # 2. Idempotency check: Re-pushing exact same version (version 1) must be ignored / 409 conflict
    resp_dup = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": cat_payload,
    })
    assert resp_dup.status_code == 409
    dup_data = resp_dup.json()
    assert dup_data["accepted"] is False
    assert dup_data["reason"] == "stale_version"
    assert dup_data["current_version"] == 1

    # 3. Re-pushing lower version (version 0) must be rejected with 409
    resp_lower = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 0,
        "payload": cat_payload,
    })
    assert resp_lower.status_code == 409
    lower_data = resp_lower.json()
    assert lower_data["accepted"] is False
    assert lower_data["reason"] == "stale_version"
    assert lower_data["current_version"] == 1

    # 4. Version bump (version 2): Must atomically replace
    cat_payload_v2 = dict(cat_payload)
    cat_payload_v2["peer_stats"]["avg_rating"] = 4.6
    resp_v2 = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 2,
        "payload": cat_payload_v2,
    })
    assert resp_v2.status_code == 200
    v2_data = resp_v2.json()
    assert v2_data["accepted"] is True
    assert v2_data["ack_id"] == "ack_dentists_v2"

    # Count should still be 1 (replaced, not duplicated)
    assert client.get("/v1/healthz").json()["contexts_loaded"]["category"] == 1


def test_context_invalid_scope():
    resp = client.post("/v1/context", json={
        "scope": "unknown_scope",
        "context_id": "item1",
        "version": 1,
        "payload": {},
    })
    assert resp.status_code == 400
    data = resp.json()
    assert data["accepted"] is False
    assert data["reason"] == "invalid_scope"


def test_tick_dynamic_composition():
    # Setup Category
    client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": {
            "slug": "dentists",
            "voice": {"tone": "peer_clinical"},
            "peer_stats": {"avg_rating": 4.5, "avg_ctr": 0.032},
            "digest": [
                {
                    "id": "d_fluoride_study",
                    "title": "3-month fluoride recall cuts caries 38%",
                    "source": "JIDA Oct 2026",
                    "summary": "2,100 patient trial shows significant caries reduction.",
                }
            ],
        },
    })

    # Setup Merchant
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001",
        "version": 1,
        "payload": {
            "merchant_id": "m_001",
            "category_slug": "dentists",
            "identity": {
                "name": "Dr. Meera's Dental Clinic",
                "owner_first_name": "Meera",
                "city": "Delhi",
                "locality": "Lajpat Nagar",
            },
            "performance": {"views": 2500, "calls": 24, "ctr": 0.024},
            "offers": [{"id": "off_1", "title": "Dental Cleaning @ ₹299", "status": "active"}],
        },
    })

    # Setup Trigger (merchant scope)
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trg_001",
        "version": 1,
        "payload": {
            "id": "trg_001",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001",
            "payload": {"top_item_id": "d_fluoride_study"},
            "suppression_key": "suppress:research:m_001",
        },
    })

    # Tick
    tick_resp = client.post("/v1/tick", json={
        "available_triggers": ["trg_001"],
    })
    assert tick_resp.status_code == 200
    actions = tick_resp.json()["actions"]
    assert len(actions) == 1

    action = actions[0]
    assert action["merchant_id"] == "m_001"
    assert action["trigger_id"] == "trg_001"
    assert action["send_as"] == "vera"
    assert "Dr. Meera" in action["body"]
    assert "JIDA Oct 2026" in action["body"]
    assert "http" not in action["body"]  # No URLs
    assert action["cta"] == "open_ended"
    assert action["suppression_key"] == "suppress:research:m_001"

    # Subsequent tick with same trigger should be suppressed
    tick_resp2 = client.post("/v1/tick", json={
        "available_triggers": ["trg_001"],
    })
    assert tick_resp2.status_code == 200
    assert len(tick_resp2.json()["actions"]) == 0


def test_tick_customer_scope():
    # Setup Category & Merchant
    client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": {"slug": "dentists", "voice": {"tone": "peer_clinical"}},
    })
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001",
        "version": 1,
        "payload": {
            "merchant_id": "m_001",
            "category_slug": "dentists",
            "identity": {"name": "Dr. Meera's Clinic"},
        },
    })
    # Setup Customer
    client.post("/v1/context", json={
        "scope": "customer",
        "context_id": "c_001",
        "version": 1,
        "payload": {
            "customer_id": "c_001",
            "identity": {"name": "Priya", "language_preference": "hi-en"},
        },
    })

    # Setup Customer Recall Trigger
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trg_customer_recall",
        "version": 1,
        "payload": {
            "id": "trg_customer_recall",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": "m_001",
            "customer_id": "c_001",
            "payload": {
                "service_due": "dental_cleaning",
                "available_slots": [
                    {"label": "Wed 5 Nov, 6pm"},
                    {"label": "Thu 6 Nov, 5pm"},
                ],
            },
            "suppression_key": "recall:c_001:2026",
        },
    })

    tick_resp = client.post("/v1/tick", json={
        "available_triggers": ["trg_customer_recall"],
    })
    assert tick_resp.status_code == 200
    actions = tick_resp.json()["actions"]
    assert len(actions) == 1
    action = actions[0]
    assert action["customer_id"] == "c_001"
    assert action["send_as"] == "merchant_on_behalf"
    assert "Priya" in action["body"]
    assert "Wed 5 Nov, 6pm" in action["body"]
    assert "Thu 6 Nov, 5pm" in action["body"]
    assert action["cta"] == "multi_choice_slot"



def test_reply_auto_reply_detection():
    conv_id = "conv_test_auto"
    auto_msg = "Thank you for contacting us! Our team will respond shortly."

    # Turn 1: Bot waits
    r1 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "from_role": "merchant",
        "message": auto_msg,
        "turn_number": 1,
    })
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["action"] == "wait"
    assert data1["wait_seconds"] == 14400

    # Turn 2: Repeated auto-reply -> Bot ends
    r2 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "from_role": "merchant",
        "message": auto_msg,
        "turn_number": 2,
    })
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["action"] == "end"


def test_reply_intent_transition():
    r = client.post("/v1/reply", json={
        "conversation_id": "conv_intent",
        "from_role": "merchant",
        "message": "Ok lets do it. Whats next?",
        "turn_number": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "send"
    body_lower = data["body"].lower()

    # Must contain actioning words
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
    assert any(w in body_lower for w in actioning)

    # Must NOT contain qualifying words
    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    assert not any(w in body_lower for w in qualifying)


def test_reply_hostile():
    r = client.post("/v1/reply", json={
        "conversation_id": "conv_hostile",
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "turn_number": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "end"


def test_reply_off_topic_gst():
    r = client.post("/v1/reply", json={
        "conversation_id": "conv_gst",
        "from_role": "merchant",
        "message": "Can you also help me with my GST filing this month?",
        "turn_number": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "send"
    assert "gst" in data["body"].lower() or "ca" in data["body"].lower() or "accountant" in data["body"].lower()
