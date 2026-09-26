# magicpin AI Challenge — Vera Merchant AI Assistant ("Vera")

An autonomous, stateful AI chatbot built with FastAPI that engages and assists local Indian merchants (and their customers) over WhatsApp. Complies with all requirements of the **magicpin AI Challenge**.

---

## 1. Overview & Framework

Vera is designed to solve real-world WhatsApp commerce challenges faced by ~100,000 local merchants across 50+ Indian cities:
- **Auto-reply pollution**: Detects canned WhatsApp Business auto-replies ("Thank you for contacting...") without burning turns.
- **Intent-handoff failures**: Seamlessly switches to action mode when a merchant commits ("let's do it") instead of asking redundant qualification questions.
- **Service+Price copy**: Uses high-converting service and price offers (e.g., "Dental Cleaning @ ₹299") instead of generic percentage discounts.
- **No external URLs**: Strictly compliant with Meta's outbound WhatsApp messaging policies.

### The 4-Context Framework
Every message is dynamically composed from 4 structured context layers:
$$\text{compose}(\text{category}, \text{merchant}, \text{trigger}, \text{customer?}) \rightarrow \text{message}$$

1. **CategoryContext**: Slow-changing vertical knowledge (Dentists, Salons, Gyms, Restaurants, Pharmacies) defining tone, vocabulary taboos, peer benchmarks, and curated digests.
2. **MerchantContext**: Business state, owner identity, active offers, performance deltas (views, calls, CTR), and customer cohort aggregates.
3. **TriggerContext**: Specific event prompting the message (e.g. `research_digest`, `recall_due`, `perf_dip`, `compliance`, `milestone_reached`, `renewal_due`).
4. **CustomerContext**: End-customer details for customer-facing touches (e.g. appointment recall slots, language preference).

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
├───────────────────────────────┬──────────────────────────────┤
│  GET  /v1/healthz             │  Liveness probe & counts     │
│  GET  /v1/metadata            │  Bot & model metadata        │
│  POST /v1/context             │  Idempotent state store      │
│  POST /v1/tick                │  4-context action composer   │
│  POST /v1/reply               │  Conversational state machine│
│  POST /v1/reset               │  State wipe (testing)        │
└───────────────────────────────┴──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 ┌──────────────────────┐             ┌─────────────────────────┐
 │   In-Memory Store    │             │      LLM Composer       │
 │   (Idempotent by     │             │  Gemini 2.5 / 2.0 Flash │
 │  scope, context_id,  │             │     (temperature=0)     │
 │       version)       │             └────────────┬────────────┘
 └──────────────────────┘                          │ Fallback on
                                                   │ timeout / 429
                                                   ▼
                                      ┌─────────────────────────┐
                                      │ Deterministic Composer  │
                                      │   (Instant Fallback)    │
                                      └─────────────────────────┘
```

### In-Memory State Store & Idempotency
- Keyed by `(scope, context_id)` across allowed scopes: `category`, `merchant`, `customer`, `trigger`.
- **Duplicate or Lower Version** (`version <= current_version`): Ignored without mutating state; returns `HTTP 409 Conflict` with `{"accepted": false, "reason": "stale_version", "current_version": cur_ver}`.
- **Higher Version** (`version > current_version`): Atomically replaces the lower version and returns `HTTP 200 OK` with an `ack_id`.
- **Invalid Scope**: Returns `HTTP 400 Bad Request`.

### Dual-Layer Composition (Gemini + Deterministic Fallback)
1. **Primary Composer**: Google Gemini (`gemini-2.5-flash` or `gemini-2.0-flash`) invoked at `temperature=0.0` with structured JSON output schema.
2. **Instant Fallback**: If the Gemini call times out (>8s), hits an `HTTP 429` rate limit, or if no API key is supplied, the bot immediately falls back to the deterministic 4-context composer (sub-millisecond execution).

---

## 3. Environment Variables

Both `bot.py` and `judge_simulator.py` support the following environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | LLM backend (`gemini`, `openai`, `anthropic`, `groq`, etc.) |
| `BOT_URL` | `http://localhost:8000` | Target URL of the bot |
| `GEMINI_API_KEY`| *(empty)* | Google Gemini API key |
| `LLM_API_KEY` | *(empty)* | Generic LLM API key (falls back to `GEMINI_API_KEY`) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model variant (`gemini-2.5-flash`, `gemini-2.0-flash`) |
| `PORT` | `8000` | HTTP port for the FastAPI server |

---

## 4. Local Installation & Development

### 1. Clone & Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
# On Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key"
export BOT_URL="http://localhost:8000"

# On Windows PowerShell
$env:GEMINI_API_KEY = "your-gemini-api-key"
$env:BOT_URL = "http://localhost:8000"
```

### 3. Start the Server
```bash
uvicorn bot:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4. Run Automated Tests
```bash
python -m pytest test_bot.py -v
```

### 5. Run the Judge Simulator
```bash
python judge_simulator.py
```

---

## 5. API Endpoints

### `GET /v1/healthz`
Liveness probe.
```json
{
  "status": "ok",
  "uptime_seconds": 120,
  "contexts_loaded": {
    "category": 5,
    "merchant": 50,
    "customer": 200,
    "trigger": 100
  }
}
```

### `GET /v1/metadata`
Bot identity and approach metadata.
```json
{
  "team_name": "Team Vera",
  "team_members": ["AI Engineer"],
  "model": "gemini (gemini-2.5-flash)",
  "approach": "Gemini temperature=0 composition with deterministic fallback + 4-context framework",
  "contact_email": "candidate@example.com",
  "version": "1.0.0",
  "submitted_at": "2026-04-26T08:00:00Z"
}
```

### `POST /v1/context`
Push category, merchant, customer, or trigger context.
```json
{
  "scope": "merchant",
  "context_id": "m_001_drmeera",
  "version": 1,
  "payload": { ... }
}
```

### `POST /v1/tick`
Simulated time tick to trigger proactive outbound actions.
```json
{
  "now": "2026-04-26T10:35:00Z",
  "available_triggers": ["trg_001_research_digest_dentists"]
}
```

### `POST /v1/reply`
Merchant or customer reply processing. Returns one of:
- `{"action": "send", "body": "...", "cta": "...", "rationale": "..."}`
- `{"action": "wait", "wait_seconds": 14400, "rationale": "..."}`
- `{"action": "end", "rationale": "..."}`

---

## 6. Deployment (Render)

A [`render.yaml`](file:///c:/Users/kashi/OneDrive/Desktop/magicpin/render.yaml) blueprint is provided.

> **Important**: The start command uses `--workers 1`:
> ```bash
> uvicorn bot:app --host 0.0.0.0 --port $PORT --workers 1
> ```
> Since the state store is held in memory, running with a single worker ensures that context pushes, ticks, and conversation turns are processed in the same memory space without race conditions or state desynchronization.
