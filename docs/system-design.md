# System design

Architecture for the Singapore Mental Health Service Navigator. Product intent, provider catalogue, `GraphState` fields, node responsibilities, KPIs, and build steps live in the [root README](../README.md) and [project plan](./project-plan.md). This document specifies **how processes, contracts, and data stores fit together**—not what the product is for.

---

## 1. Design constraints (engineering)

| Constraint | Implication |
|------------|-------------|
| Two-process local demo | Streamlit never imports the graph or holds `OPENAI_API_KEY`. All model and retrieval I/O is behind FastAPI. |
| Deterministic control flow | Edges are predicates on `GraphState` booleans. The LLM may **fill fields**; it may not **choose the next node**. |
| Fail closed on safety | Regex, timeout, malformed structured output, and empty crisis classification all converge on the hardcoded fallback payload. |
| Trace is a first-class output | The API response is invalid if `trace` is missing or if `path` is empty. The UI must not invent trace rows. |
| Curated corpus only | Retrieval never hits the open web. Ingest is a batch from `data/services.json`. |
| Ephemeral sessions (MVP) | One HTTP request = one graph run. No multi-turn memory in the graph. Streamlit may keep *display* history only. |

---

## 2. Runtime topology

Two OS processes on loopback. Eval and pytest are a third consumer of the same API (or of the compiled graph in-process for unit tests).

```mermaid
flowchart LR
  subgraph browser [Browser]
    User
  end

  subgraph uiProc ["ui/ — Streamlit :8501"]
    ChatPanel
    TracePanel
  end

  subgraph apiProc ["app/ — FastAPI + Uvicorn :8000"]
    HTTP["POST /chat/invoke"]
    Graph["Compiled LangGraph"]
    Regex["Crisis regex"]
    Rules["Tier rules"]
    Fallback["Hardcoded contacts"]
  end

  subgraph stores [Local stores]
    Chroma["ChromaDB persist dir"]
    SQLite["SQLite checkpointer / run log"]
    JSON["data/services.json"]
  end

  subgraph ext [External]
    OpenAI["OpenAI gpt-4o-mini"]
  end

  User --> ChatPanel
  ChatPanel --> HTTP
  HTTP --> Graph
  Graph --> Regex
  Graph --> Rules
  Graph --> Fallback
  Graph --> OpenAI
  Graph --> Chroma
  Graph --> SQLite
  JSON --> Chroma
  HTTP --> TracePanel
  HTTP --> ChatPanel
```

**Why this split:** Streamlit reruns the script on every widget event. Binding the graph to the UI process would re-enter model calls, leak secrets into the frontend process, and make `eval.py` depend on a browser session. FastAPI owns a single compiled graph instance (module-level or lifespan) and exposes a stable contract.

**Startup order:** ingest (if Chroma collection empty) → compile graph → bind Uvicorn → start Streamlit pointing at `http://127.0.0.1:8000`.

---

## 3. Component responsibilities

| Component | Owns | Must not own |
|-----------|------|----------------|
| [`ui/`](../ui/README.md) | Layout, chat history in `st.session_state`, spinner, rendering of `trace` | Prompts, regex, provider ranking, API keys |
| [`app/`](../app/README.md) HTTP layer | Validation, timeouts, mapping graph output → `ChatResponse`, `/health` | Widget state |
| LangGraph compiler | Node functions, conditional edges, appending to `reasoning_trace` | HTTP status codes |
| [`rag/`](../rag/README.md) | Embed + upsert, metadata `where` clause, k-NN on the **filtered** subset | Crisis detection |
| SQLite | Optional LangGraph checkpointer thread_id; optional eval run ids | Emergency phone numbers |
| OpenAI | Intent JSON + parameter JSON only | Final user-visible crisis copy |

Module sketch (implementation may rename files; the **boundaries** are the design):

```text
app/
  main.py          # FastAPI app, CORS not required for same-machine Streamlit
  schemas.py       # ChatRequest / ChatResponse / TraceEvent
  graph.py         # StateGraph compile
  state.py         # GraphState (see project plan §3.1)
  nodes/           # one module per node; no FastAPI imports
  safety.py        # regex + FALLBACK_COPY constants
  timeouts.py      # asyncio.wait_for around LLM calls
rag/
  ingest.py        # idempotent upsert by provider_id
  retrieve.py      # where-filter then query
  embed.py         # embedding function shared by ingest and query
ui/
  app.py           # Streamlit entry
  api_client.py    # httpx to /chat/invoke
```

---

## 4. HTTP contracts

### 4.1 `POST /chat/invoke`

Request (Pydantic). Extra fields rejected.

```json
{
  "query": "I am 18, have no income, and feel overwhelmed",
  "request_id": "optional-uuid"
}
```

| Field | Rules |
|-------|--------|
| `query` | Required; strip; max length bounded (e.g. 2000 chars) to cap token cost and prompt injection surface |
| `request_id` | Optional; echoed in `trace` for eval alignment |

Response: HTTP 200 for **all handled routing outcomes**, including crisis and out-of-scope. Those are product successes, not transport errors.

```json
{
  "request_id": "…",
  "message": "…user-facing text…",
  "trace": {
    "path": ["intent_gate", "param_extract", "tier_map", "retrieve", "decide"],
    "active_node": "decide",
    "is_crisis": false,
    "is_out_of_scope": false,
    "intent_confidence": 0.86,
    "parameters": { "age": 18, "budget": "Free", "urgency": "sub_acute", "primary_need": "youth_assessment" },
    "tier": "Tier 1-2",
    "citations": [
      {
        "provider_id": "chat",
        "title": "CHAT",
        "distance": 0.21,
        "metadata": { "cost_tier": "Free", "age_min": 16, "age_max": 30 }
      }
    ],
    "latency_ms": { "intent_gate": 420, "retrieve": 35, "total": 980 },
    "fallback_reason": null
  }
}
```

`citations` must be documents **actually returned by Node 4**, not model-invented names.

### 4.2 Transport errors (not routing outcomes)

| Status | When |
|--------|------|
| `422` | Empty / oversized `query` |
| `504` | Whole-graph deadline exceeded; body still includes hardcoded fallback if the timeout handler ran |
| `503` | Chroma collection missing and ingest has not run |
| `500` | Uncaught exception after fallback construction failed |

### 4.3 `GET /health`

Liveness: process up. Readiness: graph compiled **and** collection count ≥ 1 (or ≥ 8 after ingest). Streamlit should disable send until readiness is true.

---

## 5. Control-flow graph (predicates only)

Node *semantics* are in [project plan §3.2](./project-plan.md#32-node-execution-workflow). Below is the **edge algebra** the compiler must implement.

```mermaid
stateDiagram-v2
  [*] --> IntentGate

  IntentGate --> SafetyFallback: is_crisis
  IntentGate --> SafetyFallback: is_out_of_scope
  IntentGate --> ParamExtract: else

  ParamExtract --> TierMap
  TierMap --> Retrieve
  Retrieve --> Decide
  Decide --> [*]

  SafetyFallback --> [*]
```

| Edge | Predicate (evaluated in Python, not by the LLM) |
|------|--------------------------------------------------|
| IntentGate → SafetyFallback | `is_crisis or is_out_of_scope` |
| IntentGate → ParamExtract | `not is_crisis and not is_out_of_scope` |
| All other sequential edges | Unconditional |

**Ordering inside IntentGate (single node, two stages):**

1. Regex on normalised text (lowercase, collapsed whitespace). Match ⇒ set `is_crisis=True` and **skip the LLM call**.
2. Else structured LLM. If parse fails, timeout, or schema mismatch ⇒ treat as fail-closed: `is_crisis=True` **or** a dedicated `fallback_reason="llm_unavailable"` that still routes to SafetyFallback (crisis-shaped copy is acceptable; unconstrained generation is not).

Out-of-scope is **never** inferred by regex of clinical jargon alone (too many false positives on “anxiety”). It is LLM-structured plus optional keyword hints that *raise* prior, not auto-fire.

---

## 6. Sequence: three product paths

### 6.1 In-scope navigation

```mermaid
sequenceDiagram
  actor U as User
  participant S as Streamlit
  participant A as FastAPI
  participant G as LangGraph
  participant R as Regex
  participant O as OpenAI
  participant C as ChromaDB

  U->>S: submit query
  S->>A: POST /chat/invoke
  A->>G: ainvoke(state)
  G->>R: scan
  R-->>G: no match
  G->>O: intent JSON
  O-->>G: in_scope
  G->>O: extract parameters
  O-->>G: age, budget, need
  Note over G: tier rules (no LLM)
  G->>C: where filter then kNN
  C-->>G: eligible docs
  G-->>A: message + trace
  A-->>S: 200 ChatResponse
  S-->>U: chat bubble + trace panel
```

### 6.2 Crisis hard stop

```mermaid
sequenceDiagram
  actor U as User
  participant S as Streamlit
  participant A as FastAPI
  participant G as LangGraph
  participant R as Regex

  U->>S: crisis language
  S->>A: POST /chat/invoke
  A->>G: ainvoke
  G->>R: match
  Note over G,R: no OpenAI, no Chroma, no generator
  G-->>A: FALLBACK_COPY + trace.path=["intent_gate","safety_fallback"]
  A-->>S: 200
  S-->>U: verified contacts only
```

If regex misses and the LLM sets `is_crisis`, the same SafetyFallback node runs; Chroma and the decision generator still must not run.

### 6.3 Deadline / OpenAI outage

```mermaid
sequenceDiagram
  participant A as FastAPI
  participant G as LangGraph
  participant O as OpenAI

  A->>G: wait_for(ainvoke, t=T)
  G->>O: intent JSON
  O--xG: timeout
  G-->>A: SafetyFallback fallback_reason=timeout
```

`T` is a process-wide budget (target perceived latency: see project plan Step 11). Nested LLM calls share that budget; Parameter Extraction must not start if remaining time is below a floor.

---

## 7. Data stores

### 7.1 `services.json` document schema

The catalogue *contents* are listed in the [project plan](./project-plan.md#33-indexed-providers). Retrieval depends on **stable metadata names**:

| Field | Type | Used by |
|-------|------|---------|
| `provider_id` | string, unique | upsert key, citation id |
| `name` | string | display |
| `tier_labels` | string[] | e.g. `["1"]`, `["4"]`, `["1","2"]` |
| `age_min`, `age_max` | int | `where` filter; use `0` / `120` if unbounded |
| `cost_tier` | enum `Free` \| `Subsidized` \| `Private` | `where` filter |
| `urgency` | enum `crisis` \| `sub_acute` \| `routine` | filter / ranking |
| `access_pathway` | string | Node 5 copies this into `message`; model must not rewrite phone numbers inside it |
| `summary` | string | embedded text |
| `is_hard_stop_only` | bool | Tier 4 rows: never returned from Node 4; they exist in JSON for ingest completeness and for eval gold labels, not for kNN |

Embed `name + summary + access_pathway` (not phone fields as the sole vector). Keep phones in metadata **and** in `safety.py` constants; Node 6 reads constants only.

### 7.2 Chroma collection

- One collection, e.g. `sg_mh_services`.
- Distance: cosine.
- Embedding model: same function at ingest and query (OpenAI `text-embedding-3-small` is acceptable; pin the name in config). Changing the model requires a full re-ingest.
- Persist directory: local path gitignored (see [`.gitignore`](../.gitignore)); recreate via `rag/ingest.py`.
- Query path: `collection.query(where=..., query_embeddings=..., n_results=k)` with `k` small (3 is enough for eight docs). If `where` matches zero documents, return empty list and let Node 5 emit a **navigation** empty-state (e.g. national mindline as default Tier 1), never a fabricated provider.

### 7.3 SQLite

Use a **narrow** role so it does not duplicate Chroma:

| Table / store | Purpose |
|---------------|---------|
| LangGraph `SqliteSaver` (optional) | Replay a `thread_id` during debugging; **not** required for the dual-panel demo |
| `eval_runs` (optional) | Persist `request_id`, gold label, predicted provider, latencies for Step 10 |

Do not store user queries long-term in the demo; eval fixtures live as files under [`eval/`](../eval/README.md).

---

## 8. Trace protocol (UI binding)

The right panel is a **pure function of `trace`**. Suggested rows:

1. `path` as a breadcrumb (highlight `active_node`)
2. Flags: `is_crisis`, `is_out_of_scope`, `fallback_reason`
3. `intent_confidence` (hide or show “n/a” on regex-only crisis)
4. Extracted parameters (JSON, pretty-printed)
5. `tier`
6. Citations: `provider_id`, metadata chips, distance

If `is_crisis`, citations stay empty by contract. Filling them would imply retrieval ran.

Each node appends `{ "node", "t_ms", "note" }` to an internal list that serialises into `path` + `latency_ms`. Nodes must not overwrite earlier path entries.

---

## 9. Latency budget

End-to-end target is in the [project plan](./project-plan.md#step-11--system-tuning-and-ux). Split for design (not a KPI):

| Hop | Budget hint | Notes |
|-----|-------------|--------|
| Regex | < 5 ms | Always first |
| Intent LLM | ~40% of remainder | `gpt-4o-mini`, max tokens small, JSON mode |
| Extract LLM | ~40% | Skip entirely on crisis / out-of-scope |
| Tier rules | < 1 ms | |
| Chroma | ~10% | Local; warm collection |
| Decide | template fill, no LLM on MVP | Keeps tail latency and hallucination off the message |
| HTTP + Streamlit | remainder | Spinner covers this |

Node 5 should be **templated** (constraints + citation + `access_pathway`). A second generative call for “warmer tone” is out of scope and fights the latency budget.

---

## 10. Security and trust boundary

```mermaid
flowchart TB
  subgraph untrusted [Untrusted]
    Browser
    QueryText
  end

  subgraph trusted [Trusted process :8000]
    Validator
    Graph
    Key["OPENAI_API_KEY"]
    Constants["FALLBACK_COPY"]
  end

  Browser -->|HTTP JSON| Validator
  QueryText --> Validator
  Validator --> Graph
  Key --> Graph
  Constants --> Graph
```

- Streamlit may log `request_id` and display `message`; it must not log raw API keys.
- User text is only interpolated into **fixed** prompt templates as a JSON string field, never concatenated into instructions that say “ignore previous”.
- No tool-calling that can fetch URLs.
- CORS default-deny; Streamlit uses server-side `httpx`, not the browser, to call FastAPI (avoids exposing the API to arbitrary origins).

---

## 11. Test and eval seams

| Seam | Caller | What it proves |
|------|--------|----------------|
| `safety.regex` unit | [`tests/`](../tests/README.md) | Keyword hits without LLM |
| Node functions with mocked LLM | pytest | Edge predicates, empty Chroma, schema parse failure |
| `POST /chat/invoke` | pytest + httpx | Payload shape; crisis path does not call a mocked OpenAI client |
| Compiled graph in-process | [`eval/`](../eval/README.md) | Same 30 cases as HTTP, lower overhead |
| Baseline client | eval | Direct `gpt-4o-mini` chat completion, no graph |

Eval must not scrape Streamlit. The UI is out of the accuracy loop; Trace Transparency is checked by asserting `trace` keys on the API (and a thin UI test only if needed later).

---

## 12. Configuration

| Variable | Where used |
|----------|------------|
| `OPENAI_API_KEY` | Intent + extract + embeddings |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Pin; must match ingested vectors |
| `CHROMA_PATH` | Persist dir |
| `API_DEADLINE_S` | `wait_for` around `ainvoke` |
| `NAVIGATOR_API_URL` | Streamlit client, default `http://127.0.0.1:8000` |

Loaded via environment / `.env` on the **API** process. Streamlit needs only `NAVIGATOR_API_URL`.

---

## 13. What this design deliberately omits

- Auth, multi-tenant isolation, production hosting
- Streaming tokens to the chat pane (trace completeness is easier on a single JSON response)
- Multi-turn conversational memory
- Open-web or mindline.sg scraping

Those would change the trust boundary and the eval story; they are not required for the dual-panel local navigator.
