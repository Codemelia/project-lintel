# Project plan: Singapore Mental Health Service Navigator

Implementable plan for the Navigator. Product overview and public service links: [root README](../README.md).


| Field      | Value                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Build**  | Two-phase plan (14 steps)                                                                                                                     |
| **Domain** | Non-clinical service navigation (Singapore)                                                                                                   |
| **Policy** | [Singapore National Tiered Care Model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) (Tiers 1–4) |
| **Stack**  | Python, FastAPI, Streamlit, LangGraph, ChromaDB, Pydantic, SQLite                                                                             |


---

## 1. Executive summary and objectives

Help-seekers must choose among primary care polyclinics, community outreach ([CREST](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/)), youth assessment ([CHAT](https://www.chat.mentalhealth.sg/)), Family Service Centres, and acute lines ([SOS 1767](https://www.sos.org.sg/), IMH). That choice maps poorly onto unstructured lived situations.

[mindline.sg](https://mindline.sg) and related tools supply CBT-style micro-interventions and questionnaires. Official MOHT positioning treats conversational self-care bots as **CBT micro-tools**, not multi-constraint routers. They do not parse free text such as *“I am 18, have no income, and feel overwhelmed”* against National Tiered Care Model rules.

**Product objective:** a deterministic, agentic **decision engine** on LangGraph that is a **service router and preparation assistant**, not a clinician or therapist.

**Non-goals (explicit refusals):**

- Medical diagnosis
- CBT / unconstrained therapy
- Replacing crisis hotlines (crisis is **hand-off only**)

**Strategic rationale:** dropping unconstrained therapeutic bots and cultural anti-stereotyping tasks reduces liability and yields engineering metrics: Scope Adherence, Routing Precision, Safety Hand-off Success. Analogous front-door e-triage (e.g. UK NHS Limbic Access evaluations) shows pathway allocation can reduce assessment burden when the AI is **not** the treatment.

---



## 2. System architecture and technical stack

Target properties: low latency, local-first storage, and **architectural transparency** for evaluation (every recommendation must explain *why*). Component diagram, API shapes, and store responsibilities: [system-design.md](./system-design.md).


| Layer         | Technology        | Architectural rationale                                             | Code home                                                |
| ------------- | ----------------- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| Frontend UI   | Streamlit         | Fast Python UI; native dual column: chat left, decision trace right | `[ui/](../ui/README.md)`                                 |
| Backend API   | FastAPI + Uvicorn | Async REST, Pydantic validation, UI/API separation                  | `[app/](../app/README.md)`                               |
| State engine  | LangGraph         | Deterministic state machine; no unconstrained LLM loops             | `[app/](../app/README.md)`                               |
| Storage / RAG | SQLite + ChromaDB | Zero-setup local vectors + relational state; metadata-first filter  | `[rag/](../rag/README.md)`, `[data/](../data/README.md)` |


**Runtime (planned):**

- API: `localhost:8000` — primary endpoint `/chat/invoke`
- UI: `localhost:8501` — `st.columns([2, 1])`

**LLM usage:** `gpt-4o-mini` for Intent Gate and structured extraction only. Emergency contacts are **hardcoded in Python**, not retrieved from model memory.

---



## 3. LangGraph engine architecture

Single-prompt LLMs are black boxes (injection, hallucination, unsafe generation). This system is a **stateful directed graph**. Transitions are programmatic conditions, not free-form chat.

### 3.1 `GraphState` (Pydantic / `TypedDict`)

Central object carried across nodes:

```python
class GraphState(TypedDict):
    user_query: str                  # Original raw user input
    is_crisis: bool                  # Regex OR LLM Intent Gate
    is_out_of_scope: bool            # Diagnosis / CBT therapy request
    extracted_parameters: Dict       # {age, budget, urgency, primary_need, ...}
    calculated_tier: str             # "Tier 1" | "Tier 2" | "Tier 3" | "Tier 4"
    retrieved_services: List[Dict]   # Filtered ChromaDB documents
    reasoning_trace: Dict            # Node logs & confidence scores
    final_response: str              # User-facing output
```

Implement in Step 3 in `[app/](../app/README.md)`. Tests in `[tests/](../tests/README.md)`.

### 3.2 Node execution workflow

Planned graph (Step 4): `IntentClassifier → RAGRetriever → DecisionEngine → SafetyResponseGenerator`, with **conditional edges** after the Intent Gate.

#### Node 1 — Intent & Safety Gate (hybrid)

- **Regex scan** for acute self-harm terms (examples from the source plan: “die”, “suicide”, “end it”). This is a **guaranteed** high-risk path, not the only path.
- **Structured LLM** (`gpt-4o-mini`, few-shot classifier) for subtler crisis and for **scope**: CBT / diagnosis → `is_out_of_scope=True`.
- If `is_crisis=True` → **immediate hard stop** (skip Nodes 2–5 generation). Jump to Node 6.



#### Node 2 — Parameter Extraction

Structured JSON via Pydantic from unstructured text:


| Field        | Examples                                                |
| ------------ | ------------------------------------------------------- |
| Age          | `18`                                                    |
| Budget tier  | `Free`, `Subsidized`, `Private`                         |
| Primary need | Youth assessment, family support, crisis, listening ear |
| Urgency      | Sub-acute vs immediate access                           |




#### Node 3 — National Tier Mapping

Rule-based mapping onto Tiers 1–4 (not an LLM guess). Aligns with [MOH’s four-level model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/): community / self-help through hospital-intensity care. Product mapping of indexed services is in [§3.3](#33-indexed-providers).

#### Node 4 — Metadata hard-filter + RAG

Query ChromaDB documents ingested from `[data/services.json](../data/README.md)`:

1. **Strict metadata first**, e.g. `target_age.min <= age <= target_age.max` **and** `cost_tier == budget`.
2. **Then** vector similarity on the filtered subset.

This order is mandatory: it is the difference between a router and a noisy chatbot.

#### Node 5 — Decision & Explanation Engine

Build a recommendation that **binds user constraints to a provider access pathway** (how to reach the service). Populate `reasoning_trace` for the Streamlit right panel (intent confidence, active node, citations).

#### Node 6 — Safety Fallback & Boundary

On crisis or out-of-scope (and on **network timeout** — Step 11):

- **No** conversational generation
- Return **hardcoded, verified** emergency / redirect copy (SOS 1767, IMH 6389 2222, and out-of-scope redirect to navigation-only language)



### 3.3 Indexed providers

Eight primary services (Step 2 knowledge base). Schema fields: target age, cost model, urgency tier, access pathway.


| Triage tier   | Provider                                | Target demographic                    | Routing criteria                  |
| ------------- | --------------------------------------- | ------------------------------------- | --------------------------------- |
| 4 Acute       | SOS (Samaritans of Singapore — 1767)    | Suicidal ideation, acute crisis       | `is_crisis = True` → hard stop    |
| 4 Acute       | IMH Emergency (6389 2222)               | Severe psychiatric emergencies        | `is_crisis = True` → hard stop    |
| 1 General     | national mindline (1771)                | Early-stage distress, listening ear   | Sub-acute / Free / 24/7           |
| 1–2 Youth     | CHAT (Community Health Assessment Team) | Youths & young adults 16–30           | Age 16–30 / free assessment       |
| 2 Primary     | Polyclinics (tele-psychology)           | Mild-to-moderate anxiety / depression | Subsidized / GP referral          |
| 2 Community   | Family Service Centres (FSCs)           | Family distress, social support       | Means-tested / community location |
| 2 Community   | CREST teams                             | Seniors & adults needing screening    | Age 18+ / community outreach      |
| 3 Specialized | Private counselling outlets             | Immediate access, custom therapy      | Higher budget / rapid booking     |


Public context (not a substitute for curated `services.json`): [MOH mental health services](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/).

---



## 4. Phased milestone plan



### Phase 1: Architecture, data schema, and core engine



#### Step 1 — Environment and project layout

- [x] Directory structure: `[app/](../app/README.md)`, `[ui/](../ui/README.md)`, `[rag/](../rag/README.md)`, `[tests/](../tests/README.md)`, plus `[docs/](./README.md)`, `[data/](../data/README.md)`, `[eval/](../eval/README.md)`
- [ ] Virtualenv and deps: `pip install -r requirements.txt` (see `[requirements.txt](../requirements.txt)`)
- [x] Root README describes product scope ([README.md](../README.md))



#### Step 2 — Knowledge base construction

- [ ] Draft `[data/services.json](../data/README.md)` with **8** structured Singapore providers
- [ ] Fields: target age range, cost model, urgency tier, access pathways, tier label
- [ ] Ingest into ChromaDB (`[rag/](../rag/README.md)`)
- [ ] Keep emergency phone numbers as structured fields **and** as hardcoded Node 6 constants (defence in depth)



#### Step 3 — State schema and Intent Gate

- [ ] `GraphState` as in [§3.1](#31-graphstate-pydantic--typeddict)
- [ ] Few-shot Intent Classifier prompt (in-scope navigation vs crisis vs out-of-scope)
- [ ] Regex keyword fallback for acute self-harm terms — must fire even if the LLM is down
- [ ] Unit tests: crisis true-positives, benign false-positive checks, diagnosis/CBT refusals



#### Step 4 — Decision engine and LangGraph workflow

- [ ] Assemble nodes: IntentClassifier → RAGRetriever → DecisionEngine → SafetyResponseGenerator
- [ ] Conditional edges: crisis / out-of-scope → Node 6; else continue pipeline
- [ ] Persist `reasoning_trace` at every hop



#### Step 5 — FastAPI backend integration

- [ ] Wrap graph in REST: `POST /chat/invoke`
- [ ] Async processing; Pydantic request/response models
- [ ] pytest: node execution, payload shapes, crisis path does not call the generator



#### Steps 6–7 — Streamlit dual-panel interface

- [ ] `st.columns([2, 1])`
- [ ] **Left:** chat
- [ ] **Right:** intent confidence, active router node, RAG citations
- [ ] Trace Transparency KPI: empty right panel = failure



### Phase 2: Evaluation, UX polish, and presentation



#### Steps 8–9 — Benchmark dataset

- [ ] Synthesize **30** mock scenarios in `[eval/](../eval/README.md)`:
  - In-scope navigation (15): age / budget / need → correct provider
  - Acute crisis (8): hard-stop to SOS / IMH (no chatty empathy loop)
  - Out-of-scope (7): CBT, diagnosis, “be my therapist” → refuse + redirect
- [ ] Gold labels: expected `is_crisis` / `is_out_of_scope`, expected provider id, expected tier



#### Step 10 — Automated evaluation pipeline

- [ ] `eval.py`: Baseline = direct GPT-4o-mini (no graph) vs Navigator
- [ ] Report: Scope Adherence %, Safety Handoff Success %, Navigation Accuracy %
- [ ] Store raw outputs for the deck (Step 14)



#### Step 11 — System tuning and UX

- [ ] Prompt/graph latency target: **under 2 seconds** perceived (spinners on Streamlit)
- [ ] Hardcode fallback contacts for emergency nodes on **timeouts**
- [ ] Use `gpt-4o-mini` only where specified (latency risk in [§6](#6-risk-management--contingency))



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
- [ ] Deck: architectural safety (gates, hard stops, no unconstrained loops)
- [ ] Deck: empirical benchmark results vs baseline
- [ ] Export final reports from `[eval/](../eval/README.md)`

---



## 5. Evaluation strategy and KPIs

**Baseline:** unconstrained GPT-4o-mini (no state routing) on the same 30 queries.


| Metric                       | Target | Measurement                                                                                     |
| ---------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| **Safety Hand-off Success**  | 100%   | Zero failures on acute crisis: hard stop + emergency contacts; **no** conversational generation |
| **Scope Adherence Rate**     | > 90%  | Out-of-scope (CBT, diagnosis) correctly refused and redirected to non-clinical navigation       |
| **Service Routing Accuracy** | > 85%  | Age, budget, urgency match the **exact eligible** provider in `services.json`                   |
| **Trace Transparency**       | 100%   | Right-hand panel populated with verified RAG citations and router metadata every time           |


Failure modes to log (for tuning Steps 10–11):

- Crisis missed by regex but caught by LLM (acceptable) vs missed by both (blocker)
- Eligible provider filtered out by overly strict metadata
- Baseline “helpfully” giving therapy instead of routing (expected baseline failure; Navigator must not copy it)

---



## 6. Risk management and contingency


| Risk                      | Why it matters                                             | Mitigation                                                                                                                                                                                                     |
| ------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latency**               | Multistep LangGraph + two model calls can exceed UX budget | `gpt-4o-mini` for classification; Streamlit spinners; keep Node 3 rule-based; Step 11 latency pass                                                                                                             |
| **Outdated contact data** | Singapore hotlines can change                              | Hardcode contacts in Python Node 6; do **not** trust LLM memory; re-verify against [MOH directory](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/) before freeze |
| **Safety bypasses**       | Models miss subtle crisis language                         | Hybrid regex **and** LLM; crisis edge skips generator; timeout fallback still returns hardcoded numbers                                                                                                        |
| **Scope creep**           | Pressure to “just be a bit therapeutic”                    | Out-of-scope flag is a first-class graph field; KPI is refusal rate, not empathy score                                                                                                                         |
| **RAG noise**             | Wrong provider despite good intent                         | Metadata filter **before** vectors; 8-document curated set (not an open web crawl)                                                                                                                             |


---



## 7. Definition of done

Work is complete when all of the following hold:

- [ ] Dual-panel UI + FastAPI graph path work locally (Step 13).
- [ ] Crisis path never generates free-form therapy; contacts are hardcoded.
- [ ] Eval report exists for 30 cases vs baseline; Safety Hand-off is 100% on the crisis slice (or residual misses are documented as blockers, not ignored).
- [ ] Deck covers problem, safety architecture, and numbers.
- [ ] README and this plan stay consistent with each other.

---



## 8. Document map


| Artefact                         | Location                                                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Product README                   | [../README.md](../README.md)                                                                              |
| System design                    | [system-design.md](./system-design.md)                                                                    |
| Docs index                       | [README.md](./README.md)                                                                                  |
| National strategy                | [go.gov.sg/mental-health-strategy](https://go.gov.sg/mental-health-strategy)                              |
| Tiered Care Model announcement   | [MOH newsroom](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) |
| Stepped / tiered care (academic) | [BMC Global and Public Health](https://link.springer.com/article/10.1186/s44263-025-00196-0)              |


