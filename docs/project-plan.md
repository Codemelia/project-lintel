# Project plan: Singapore Mental Health Service Navigator

Implementable plan for the Navigator. Product overview and public service links: [root README](../README.md).


| Field      | Value                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Build**  | Two-phase plan (14 steps)                                                                                                                     |
| **Domain** | Non-clinical service navigation (Singapore)                                                                                                   |
| **Policy** | [Singapore National Tiered Care Model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) (Tiers 1–4) |
| **Stack**  | Python, FastAPI, Streamlit (chatbot UI), LangGraph, `gpt-4o-mini` (intent/extract JSON), ChromaDB + MiniLM, Pydantic, SQLite                    |


---



## 1. Executive summary and objectives

Help-seekers must choose among primary care polyclinics, community outreach ([CREST](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/)), youth assessment ([CHAT](https://www.chat.mentalhealth.sg/)), Family Service Centres, and acute lines ([SOS 1767](https://www.sos.org.sg/), IMH). That choice maps poorly onto unstructured lived situations.

[mindline.sg](https://mindline.sg) and related tools supply CBT-style micro-interventions and questionnaires. Official MOHT positioning treats conversational self-care bots as **CBT micro-tools**, not multi-constraint routers. They do not parse free text such as *“I am 18, have no income, and feel overwhelmed”* against National Tiered Care Model rules.

**Product objective:** a public-facing **service navigator chatbot**: chat UI in front of a deterministic LangGraph **decision engine**. It is a **service router and preparation assistant**, not a clinician or therapist.

**Non-goals (explicit refusals):**

- Medical diagnosis
- CBT / unconstrained therapy
- Replacing crisis hotlines (crisis is **hand-off only**)

**Strategic rationale:** dropping unconstrained therapeutic bots reduces liability and yields engineering metrics: Scope Adherence, Routing Precision, Safety Hand-off Success, and Trace Transparency. Analogous front-door e-triage (e.g. UK NHS Limbic Access evaluations) shows pathway allocation can reduce assessment burden when the AI is **not** the treatment.

---



## 2. System architecture and technical stack

Target properties: low latency, **architectural transparency** (every recommendation must explain *why*), and a **constrained chatbot** (models fill JSON fields; Python chooses edges). Component diagram, API shapes, and store responsibilities: [system-design.md](./system-design.md).

MVP runs as a **two-process app** (same contract when hosted): Streamlit (`st.columns([2, 1])`) never holds model credentials; FastAPI owns the graph and the OpenAI client.


| Layer                          | Technology                                                                 | Architectural rationale                                                                                          | Code home                                                          |
| ------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Frontend UI                    | Streamlit chatbot                                                          | Dual column: chat left, decision trace right; no secrets in the UI process                                       | `[ui/](../ui/README.md)`                                           |
| Backend API                    | FastAPI + Uvicorn                                                          | Async REST, Pydantic validation, UI/API separation                                                               | `[app/](../app/README.md)`                                         |
| State / routing engine         | LangGraph + `get_node_model()`                                             | Deterministic state machine; no unconstrained LLM loops; models fill fields only                                 | `[app/](../app/README.md)`, `app/graph/models.py`                  |
| Intent & extract               | `gpt-4o-mini` structured JSON (`OPENAI_BASE_URL` optional)                 | Slot-filling for Nodes 1–2; regex still skips the API on first-person crisis                                     | `app/graph/models.py` (Nodes 1–2)                                  |
| Storage / RAG                  | SQLite + ChromaDB + MiniLM                                                 | Local catalogue vectors; metadata-first filter                                                                   | `[rag/](../rag/README.md)`, `[data/](../data/README.md)`           |


**Runtime (planned):**

- API: `localhost:8000` — primary endpoint `/chat/invoke` (same contract when publicly hosted)
- UI: `localhost:8501` — Streamlit chatbot, `st.columns([2, 1])`
- Nodes 1–2: `gpt-4o-mini` via `OPENAI_API_KEY` (optional `OPENAI_BASE_URL` for a private/regional endpoint)
- Catalogue embeddings: local MiniLM (`EMBEDDING_MODEL`)

**Data governance:** This is a **public-facing** chatbot. Help-seeker text for in-scope and out-of-scope paths is sent to the configured OpenAI-compatible endpoint as structured JSON (`trace.egress=true`). First-person crisis regex never calls the model (`egress=false`). Catalogue embeddings stay local. Emergency contacts remain **hardcoded in Python**. Nodes 3–5 stay rule-based / RAG / templates (no generative therapy). Before real users: privacy notice, retention limits, and a processor agreement for the LLM vendor. Singapore [PDPA](https://www.pdpc.gov.sg/overview-of-pdpa/the-legislation/personal-data-protection-act) applies to the **operator**, not to where a model binary is installed.

---



## 3. LangGraph engine architecture

Single-prompt LLMs are black boxes (injection, hallucination, unsafe generation). This system is a **stateful directed graph**. Transitions are programmatic conditions, not free-form chat.

### 3.1 `GraphState` (Pydantic / `TypedDict`)

Central object carried across nodes:

```python
class GraphState(TypedDict):
    user_query: str                  # Original raw user input
    is_crisis: bool                  # Regex OR gpt-4o-mini Intent Gate
    is_out_of_scope: bool            # Diagnosis / CBT therapy request
    extracted_parameters: Dict       # {age, cost_model, urgency_level, category, ...}
    calculated_tier: str             # "Tier 1" | "Tier 2" | "Tier 3" | "Tier 4"
    retrieved_services: List[Dict]   # Filtered ChromaDB documents (by service_id)
    reasoning_trace: Dict            # Node logs & confidence scores
    final_response: str              # User-facing output
```

Implement in Step 3 in `[app/](../app/README.md)` (`state.py`, `app/graph/models.py`). Tests in `[tests/](../tests/README.md)`.

### 3.2 Node execution workflow

Planned graph (Step 4): `IntentClassifier → RAGRetriever → DecisionEngine → SafetyResponseGenerator`, with **conditional edges** after the Intent Gate.

#### Node 1 — Intent & Safety Gate (hybrid)

Ordered stages (Python predicates still choose the next node; models only **fill fields**):

1. **Regex scan** for **first-person** self-harm (`kill myself`, `want to die`, `end my life`, …). Match ⇒ `is_crisis=True`, **no model call**. Bare `suicide` / third-person helper phrasing is **not** a regex hard-stop.
2. **Structured classifier** via `get_node_model("intent")`: `gpt-4o-mini` JSON (`in_scope` vs `crisis` vs `out_of_scope`, plus an uncalibrated confidence score).
3. If `is_crisis=True` → **immediate hard stop** (skip Nodes 2–5 generation). Jump to Node 6.

Regex is **first-person self-harm only**. Bare `suicide` / helper queries (“my friend is suicidal”) do **not** skip the model; they stay in-scope navigation unless the model labels crisis.

#### Confidence scores (policy)

JSON `confidence` from `gpt-4o-mini` is **uncalibrated**. It is not a clinical probability, not a second-look gate, and is not copied from any paper’s threshold. Log it on `reasoning_trace` for eval. Crisis **label** (or regex) still goes to Node 6 at any score.

| Band | Role in this app |
|------|------------------|
| Regex or `is_crisis=true` | Node 6 **regardless of score**. Do not wait for a high score to hand off. |
| `confidence` on the trace | Observability only. |
| **0.55 diagnostic suppressors** | **Out of scope.** MATRIX-style “hide diagnosis below 0.55” must not suppress navigation. |
| Precision/recall ~95% or Safety Hand-off 100% | **Step 10 KPIs**, not a runtime cutoff. |

#### Model client (`app/graph/models.py`)

Nodes 1 and 2 call `get_node_model(task_type)`, which returns `OpenAIStructuredClient` (`gpt-4o-mini`, optional `OPENAI_BASE_URL`). Record `model_backend` (`regex` \| `cloud`) and `egress` (`false` only on regex crisis).

#### Node 2 — Parameter Extraction

Structured JSON via Pydantic from unstructured text. Same client: `get_node_model("extract")`. No phone numbers from the model.


| Field          | Catalogue enum / examples                                                                 |
| -------------- | ----------------------------------------------------------------------------------------- |
| Age            | `18` (filter: `age_min` ≤ age ≤ `age_max`)                                                |
| Cost model     | `Free`, `Subsidized`, `Private`, `Variable` (matches `cost_model` in `services.json`)     |
| Category       | One of the six `category` values in [`services.schema.json`](../data/schemas/services.schema.json) |
| Urgency        | `routine`, `sub-acute`, `acute`, `emergency` (matches `urgency_level`)                    |




#### Node 3 — National Tier Mapping

Rule-based mapping onto Tiers 1–4 (not an LLM guess). Aligns with [MOH’s four-level model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/): community / self-help through hospital-intensity care. Product mapping of indexed services is in [§3.3](#33-indexed-providers).

#### Node 4 — Metadata hard-filter + RAG

Query ChromaDB documents ingested from [`data/services.json`](../data/services.json) (schema: [`data/schemas/services.schema.json`](../data/schemas/services.schema.json)):

1. **Strict metadata first**, e.g. `age_min <= age <= age_max` **and** `cost_model` matches extracted cost **and** `is_hard_stop_only == false`.
2. **Then** vector similarity on the filtered subset.

This order is mandatory: it is the difference between a router and a noisy chatbot.

#### Node 5 — Decision & Explanation Engine

Build a recommendation that **binds user constraints to a provider access pathway** (how to reach the service). Populate `reasoning_trace` for the Streamlit right panel (intent confidence, active node, citations).

#### Node 6 — Safety Fallback & Boundary

On crisis or out-of-scope (and on **network timeout** — Step 11):

- **No** conversational generation
- Return **hardcoded, verified** emergency / redirect copy (SOS 1767, IMH 6389 2222, and out-of-scope redirect to navigation-only language)



### 3.3 Indexed providers

Eight curated services in [`data/services.json`](../data/services.json). Contract: [`data/schemas/services.schema.json`](../data/schemas/services.schema.json) (`LintelServiceCatalog`). Citations and eval gold labels use `service_id` (`^sg-[a-z0-9-]+$`).

**Schema fields (required unless noted):**

| Field | Type / enum | Used by |
| ----- | ----------- | ------- |
| `service_id` | string, unique | Chroma upsert key, citation id, eval gold |
| `name` | string | Display |
| `tier_labels` | `["1"…"4"]`, min 1 | Tier mapping / ranking |
| `category` | six-value enum (see schema) | Metadata filter / extracted need |
| `age_min`, `age_max` | int 0–120 inclusive | `where` filter |
| `eligibility_criteria` | string[], min 1 | Explanation / trace |
| `cost_model` | `Free` \| `Subsidized` \| `Private` \| `Variable` | `where` filter |
| `urgency_level` | `routine` \| `sub-acute` \| `acute` \| `emergency` | Filter / ranking |
| `summary` | string, max 300 | Embedded text |
| `access_pathway` | string | Node 5 copies this; do not rewrite phones |
| `contact` | object: `website`, `phone`, `operating_hours` (string or null); `location_type` (enum) | Source JSON; flatten to scalars on Chroma upsert |
| `is_hard_stop_only` | bool | Tier 4 rows: ingest for completeness / gold labels; **never** return from Node 4 kNN |

Embed `name + summary + access_pathway`. Keep phones in `contact.phone` **and** in Node 6 Python constants.

| `service_id` | Tiers | Provider | Age | `cost_model` | `urgency_level` | Routing |
| ------------ | ----- | -------- | --- | ------------ | --------------- | ------- |
| `sg-sos-01` | 4 | Samaritans of Singapore (SOS) | 0–120 | Free | emergency | `is_crisis` → hard stop; `is_hard_stop_only` |
| `sg-imh-emergency-01` | 4 | IMH Emergency Services | 0–120 | Subsidized | emergency | `is_crisis` → hard stop; `is_hard_stop_only` |
| `sg-mindline-01` | 1 | National Mindline (1771) | 0–120 | Free | routine | Free / 24/7 listening ear |
| `sg-chat-01` | 1–2 | CHAT | 16–30 | Free | routine | Youth assessment; walk-in / video |
| `sg-polyclinic-telepsych-01` | 2 | Polyclinic primary care & tele-psychology | 7–120 | Subsidized | routine | GP booking; citizens / PRs |
| `sg-fsc-01` | 2 | Family Service Centres | 0–120 | Subsidized | routine | Family / social casework by postal code |
| `sg-crest-01` | 1–2 | CREST | 18–120 | Free | routine | Adult / senior outreach via AIC |
| `sg-private-counseling-01` | 3 | Private counseling & psychotherapy | 5–120 | Private | sub-acute | Rapid self-book; no public wait |

Public context (not a substitute for the curated file): [MOH mental health services](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/).

---



## 4. Phased milestone plan



### Phase 1: Architecture, data schema, and core engine



#### Step 1 — Environment and project layout

- [x] Directory structure: `[app/](../app/README.md)`, `[ui/](../ui/README.md)`, `[rag/](../rag/README.md)`, `[tests/](../tests/README.md)`, plus `[docs/](./README.md)`, `[data/](../data/README.md)`, `[eval/](../eval/README.md)`
- [x] Virtualenv and deps: `pip install -r requirements.txt` (see `[requirements.txt](../requirements.txt)`)
- [x] Root README describes product scope ([README.md](../README.md))



#### Step 2 — Knowledge base construction

- [x] Draft [`data/services.json`](../data/services.json) with **8** structured Singapore providers
- [x] JSON Schema: [`data/schemas/services.schema.json`](../data/schemas/services.schema.json) (`service_id`, nested `contact`, `cost_model`, `urgency_level`, `is_hard_stop_only`, …)
- [x] `services.json` validates against the schema (nested `contact.website` / `phone` / `operating_hours` / `location_type`)
- [x] Ingest into ChromaDB (`[rag/](../rag/README.md)`): upsert by `service_id`; flatten `contact` for metadata; exclude `is_hard_stop_only` from Node 4 queries (`[rag/retrieve.py](../rag/retrieve.py)`)
- [x] Keep emergency phones as structured fields **and** as hardcoded Node 6 constants (defence in depth): SOS `1767`, IMH `6389 2222` (`[app/safety.py](../app/safety.py)`)



#### Step 3 — State schema and Intent Gate

- [x] `GraphState` as in [§3.1](#31-graphstate-pydantic--typeddict) (`[app/state.py](../app/state.py)`)
- [x] Implement `get_node_model(task_type)` in `app/graph/models.py` (`OpenAIStructuredClient` / `gpt-4o-mini`)
- [x] Intent classifier: regex first, else one OpenAI JSON call (`classify_intent`)
- [x] Regex keyword fallback for acute self-harm terms — must fire even if OpenAI is down (`[app/safety.py](../app/safety.py)` `crisis_regex_match`)
- [x] `egress` flag: `false` on regex crisis; `true` when the utterance is sent to OpenAI
- [ ] Evaluation script in `[eval/](../eval/README.md)`: unconstrained `gpt-4o-mini` vs Navigator on the 30 benchmark cases (may run against a stub set until Steps 8–9 freeze gold labels)
- [x] Unit tests: crisis true-positives, benign false-positive checks, diagnosis/CBT refusals (mocked OpenAI); factory returns cloud client (`[tests/test_step3.py](../tests/test_step3.py)`)



#### Step 4 — Decision engine and LangGraph workflow

- [ ] Assemble nodes: IntentClassifier → RAGRetriever → DecisionEngine → SafetyResponseGenerator
- [ ] Conditional edges: crisis / out-of-scope → Node 6; else continue pipeline
- [ ] Persist `reasoning_trace` at every hop



#### Step 5 — FastAPI backend integration

- [ ] Wrap graph in REST: `POST /chat/invoke`
- [ ] Async processing; Pydantic request/response models
- [ ] pytest: node execution, payload shapes, crisis path does not call the generator



#### Steps 6–7 — Streamlit dual-panel chatbot

- [ ] `st.columns([2, 1])`
- [ ] **Left:** chatbot
- [ ] **Right:** intent confidence, active router node, RAG citations
- [ ] Trace Transparency KPI: empty right panel = failure



### Phase 2: Evaluation, UX polish, and presentation



#### Steps 8–9 — Benchmark dataset

- [ ] Synthesize **30** mock scenarios in `[eval/](../eval/README.md)`:
  - In-scope navigation (15): age / `cost_model` / need → correct `service_id`
  - Acute crisis (8): hard-stop to SOS / IMH (no chatty empathy loop)
  - Out-of-scope (7): CBT, diagnosis, “be my therapist” → refuse + redirect
- [ ] Gold labels: expected `is_crisis` / `is_out_of_scope`, expected `service_id`, expected tier



#### Step 10 — Automated evaluation pipeline

- [ ] `eval.py` reports **two** configurations on the same 30 gold cases:
  1. Pure unconstrained `gpt-4o-mini` (no graph) — **baseline**
  2. Navigator chatbot (regex + graph + `gpt-4o-mini` for intent/extract)
- [ ] Report: Scope Adherence %, Safety Handoff Success %, Navigation Accuracy %, Trace Transparency
- [ ] Store raw outputs for the deck (Step 14)



#### Step 11 — System tuning and UX

- [ ] Prompt/graph latency target: **under 2 seconds** perceived (spinners on Streamlit)
- [ ] Hardcode fallback contacts for emergency nodes on **timeouts**



#### Step 12 — Demo scripting and diagrams

- [ ] High-resolution architecture diagrams (graph + dual UI)
- [ ] **3-minute** demo script:
  - [ ] Normal service search (e.g. 18, no income, overwhelmed → CHAT / free pathway)
  - [ ] Crisis hand-off (hard stop, verified numbers)
  - [ ] Live reasoning trace on the right panel



#### Step 13 — End-to-end dry runs

- [ ] FastAPI `localhost:8000` + Streamlit `localhost:8501`
- [ ] Full integration test of in-scope, crisis, out-of-scope
- [ ] Freeze application code (docs/deck only after freeze)



#### Step 14 — Slide deck and submission freeze

- [ ] Deck: problem statement (ecosystem cognitive load)
- [ ] Deck: architectural safety (gates, hard stops, structured JSON only, no unconstrained loops)
- [ ] Deck: empirical benchmark results vs baseline (two configs)
- [ ] Export final reports from `[eval/](../eval/README.md)`

---



## 5. Evaluation strategy and KPIs

**Two-way comparison** on the same 30 queries (Step 10): unconstrained `gpt-4o-mini` (baseline) vs Navigator chatbot.


| Metric                              | Target | Measurement                                                                                          |
| ----------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| **Safety Hand-off Success**         | 100%   | Zero failures on acute crisis: hard stop + emergency contacts; **no** conversational generation      |
| **Scope Adherence Rate**            | > 90%  | Out-of-scope (CBT, diagnosis) correctly refused and redirected to non-clinical navigation            |
| **Service Routing Accuracy**        | > 85%  | Age, `cost_model`, and `urgency_level` match the **exact eligible** `service_id` in `services.json`  |
| **Trace Transparency**              | 100%   | Right-hand panel populated with verified RAG citations and router metadata every time                |

A literature-style **0.95 precision/recall floor** is this table. Model JSON `confidence` is logged only. Crisis hand-off does not require a high confidence score.


Failure modes to log (for tuning Steps 10–11):

- Crisis missed by regex but caught by `gpt-4o-mini` (acceptable) vs missed by regex and the model (blocker)
- Eligible provider filtered out by overly strict metadata
- Baseline “helpfully” giving therapy instead of routing (expected baseline failure; Navigator must not copy it)

---



## 6. Risk management and contingency


| Risk                      | Why it matters                                             | Mitigation                                                                                                                                                                                                     |
| ------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latency**               | Multistep LangGraph + two OpenAI calls can exceed UX budget | Streamlit spinners; keep Node 3 rule-based; templated Node 5; timeout → hardcoded fallback; Step 11 latency pass                                                                                              |
| **Unconstrained chatbot** | Mentors / users treat a chat UI as a therapist             | Graph refuses CBT/diagnosis; Node 5 templates only; eval vs unconstrained baseline                                                                                                                             |
| **OpenAI outage / timeout** | Nodes 1–2 cannot classify                                  | Regex still fires first; `wait_for` fail-closed returns hardcoded numbers                                                                                                                                      |
| **PDPA / public users**   | Real help-seeker text is personal data                     | Privacy notice + retention; optional private `OPENAI_BASE_URL`; MiniLM for RAG; no long-term query store in MVP; legal review before production traffic                                                         |
| **Outdated contact data** | Singapore hotlines can change                              | Hardcode contacts in Python Node 6; do **not** trust LLM memory; re-verify against [MOH directory](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/) before freeze |
| **Safety bypasses**       | Models miss subtle crisis language                         | Hybrid regex **and** structured intent; crisis edge skips generator; timeout fallback still returns hardcoded numbers                                                                                           |
| **Scope creep**           | Pressure to “just be a bit therapeutic”                    | Out-of-scope flag is a first-class graph field; KPI is refusal rate, not empathy score                                                                                                                         |
| **RAG noise**             | Wrong provider despite good intent                         | Metadata filter **before** vectors; 8-document curated set (not an open web crawl)                                                                                                                             |


---



## 7. Definition of done

Work is complete when all of the following hold:

- [ ] Dual-panel chatbot + FastAPI graph path work locally (Step 13).
- [ ] Crisis path never generates free-form therapy; contacts are hardcoded.
- [ ] Eval report exists for 30 cases across two configs (baseline / Navigator); Safety Hand-off is 100% on the crisis slice (or residual misses are documented as blockers, not ignored).
- [ ] Deck covers problem, safety architecture, and numbers.
- [ ] README and this plan stay consistent with each other.

---



## 8. Document map


| Artefact                         | Location                                                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Product README                   | [../README.md](../README.md)                                                                              |
| System design                    | [system-design.md](./system-design.md)                                                                    |
| Service catalogue                | [../data/services.json](../data/services.json)                                                            |
| Catalogue JSON Schema            | [../data/schemas/services.schema.json](../data/schemas/services.schema.json)                              |
| Docs index                       | [README.md](./README.md)                                                                                  |
| National strategy                | [go.gov.sg/mental-health-strategy](https://go.gov.sg/mental-health-strategy)                              |
| Tiered Care Model announcement   | [MOH newsroom](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) |
| Stepped / tiered care (academic) | [BMC Global and Public Health](https://link.springer.com/article/10.1186/s44263-025-00196-0)              |


