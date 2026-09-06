# Singapore Mental Health Service Navigator

A **non-clinical service router** for Singapore’s mental health ecosystem. Help-seekers describe their situation in free text; a deterministic [LangGraph](https://langchain-ai.github.io/langgraph/) engine maps constraints to the [National Tiered Care Model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) (Tiers 1–4) and returns a traceable provider pathway—not therapy, diagnosis, or CBT.

Scope, architecture, and milestones are in [`docs/project-plan.md`](./docs/project-plan.md). Process topology, API contracts, and diagrams: [`docs/system-design.md`](./docs/system-design.md).

> **Safety first.** This app is not a medical provider. Acute crisis queries must hard-stop and hand off to verified contacts: [SOS 1767](https://www.sos.org.sg/), [IMH Emergency / helpline 6389 2222](https://www.imh.com.sg/), or emergency services (995 / 999). See [If you need help now](#if-you-need-help-now).

---

## Contents

- [Problem](#problem)
- [What this system does](#what-this-system-does)
- [Policy frame](#policy-frame)
- [Architecture](#architecture) — [`docs/system-design.md`](./docs/system-design.md)
- [Knowledge base](#knowledge-base)
- [Evaluation targets](#evaluation-targets)
- [Repository layout](#repository-layout)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Build phases](#build-phases)
- [If you need help now](#if-you-need-help-now)
- [References](#references)

---

## Problem

Help-seekers in Singapore face high cognitive load when choosing among polyclinics, [CREST](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/) outreach teams, [CHAT](https://www.chat.mentalhealth.sg/) youth assessment, Family Service Centres (FSCs), and crisis lines. Existing digital tools such as [mindline.sg](https://mindline.sg) (including Wysa-style CBT micro-interventions and PHQ-9 / GAD-7 questionnaires) are valuable for self-care, but they are not **multi-constraint triage engines**. They do not reliably take unstructured input such as *“I am 18, have no income, and feel overwhelmed”* and route it against National Tiered Care Model rules.

The result: functional silos, decision fatigue, and weak measurability of routing safety.

## What this system does

The Navigator is a **transparent preparation assistant and service router**:

1. **Intent & Scope Gate** — regex plus structured LLM classification. Crisis or out-of-scope requests (diagnosis, CBT therapy) never enter conversational generation.
2. **Parameter extraction** — age, budget tier (`Free` / `Subsidized` / `Private`), urgency, and primary need from free text.
3. **Tier mapping** — rule-based assignment to Tiers 1–4.
4. **Hard-filtered RAG** — metadata filters on curated Singapore providers **before** vector search.
5. **Explainable recommendation** — user-facing pathway plus a live decision trace (intent, active node, citations).

Strategic focus is borrowed from front-door e-triage (for example UK NHS evaluations of Limbic Access): reduce clinician assessment burden by allocating pathways, not by acting as a therapist. Liability is bounded by refusing unconstrained therapeutic bots.

Measurable engineering KPIs: **Scope Adherence Rate**, **Routing Precision / Navigation Accuracy**, and **Safety Hand-off Success**. Full definitions are in the [evaluation section of the project plan](./docs/project-plan.md#5-evaluation-strategy--kpis).

## Policy frame

Singapore’s [National Mental Health and Well-Being Strategy](https://go.gov.sg/mental-health-strategy) organises services into four tiers by need severity and intervention intensity (stepped / tiered care). This product implements **navigation against that model**; it does not replace clinical assessment.

| Tier | Role in this product | Example indexed services |
|------|----------------------|--------------------------|
| **1 — General** | Early distress, listening ear, 24/7 / free | [national mindline 1771](https://www.moh.gov.sg/newsroom/national-mindline-1771-to-provide--round-the-clock-support-for-mental-health/) |
| **1–2 — Youth** | Free youth assessment | [CHAT](https://www.chat.mentalhealth.sg/) (ages 16–30) |
| **2 — Primary / community** | Mild–moderate need, community support | Polyclinic tele-psychology, FSCs, [CREST](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/) (18+) |
| **3 — Specialized** | Faster access, private budget | Private counselling outlets |
| **4 — Acute** | Suicidal ideation, psychiatric emergency | [SOS 1767](https://www.sos.org.sg/), [IMH](https://www.imh.com.sg/) `6389 2222` — **hard stop only** |

Official public directory: [MOH — Mental health services](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/).

## Architecture

Modular Python stack, low latency, high transparency for evaluation. **Diagrams, HTTP contracts, store split, and control-flow predicates:** [`docs/system-design.md`](./docs/system-design.md).

| Layer | Technology | Role |
|-------|------------|------|
| Frontend | [Streamlit](https://streamlit.io/) | Dual column: chat (left) and live reasoning trace (right) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | REST orchestration, [Pydantic](https://docs.pydantic.dev/) validation |
| State engine | [LangGraph](https://langchain-ai.github.io/langgraph/) | Deterministic graph: Intent Gate → RAG → Hard Stop / Response Generator |
| Storage / RAG | SQLite + [ChromaDB](https://www.trychroma.com/) | Local vectors + relational state; curated provider metadata |

**Graph nodes** (predicates and sequence diagrams in [system design](./docs/system-design.md); node roles in the [project plan](./docs/project-plan.md#3-langgraph-engine-architecture)):

1. Intent & Safety Gate (hybrid regex + `gpt-4o-mini`)
2. Parameter Extraction
3. National Tier Mapping
4. Metadata hard-filter + RAG
5. Decision & Explanation
6. Safety Fallback & Boundary (hardcoded emergency contacts)

Conditional edges prevent unconstrained LLM loops. Crisis / out-of-scope paths skip generation entirely.

## Knowledge base

Eight primary providers spanning Tiers 1–4, stored as structured documents (planned path: [`data/services.json`](./data/README.md)) and ingested into ChromaDB via [`rag/`](./rag/README.md).

| Tier | Provider | Routing criteria |
|------|----------|------------------|
| 4 Acute | SOS (1767) | `is_crisis = True` → hard stop |
| 4 Acute | IMH Emergency (6389 2222) | `is_crisis = True` → hard stop |
| 1 General | national mindline (1771) | Sub-acute / free / 24/7 |
| 1–2 Youth | CHAT | Age 16–30 / free assessment |
| 2 Primary | Polyclinics (tele-psychology) | Subsidized / GP referral |
| 2 Community | Family Service Centres | Means-tested / community location |
| 2 Community | CREST teams | Age 18+ / outreach |
| 3 Specialized | Private counselling | Higher budget / rapid booking |

## Evaluation targets

Benchmark: **30** synthetic queries (15 in-scope navigation, 8 acute crisis, 7 out-of-scope). Compare this Navigator against unconstrained **GPT-4o-mini** (no state routing). Pipeline: [`eval/`](./eval/README.md).

| Metric | Target | Method |
|--------|--------|--------|
| Safety Hand-off Success | **100%** | Crisis cases must hard-stop to emergency contacts; no conversational generation |
| Scope Adherence Rate | **> 90%** | Out-of-scope (CBT / diagnosis) refused and redirected to non-clinical navigation |
| Service Routing Accuracy | **> 85%** | Age, budget, and urgency match the eligible provider in `services.json` |
| Trace Transparency | **100%** | Every output fills the right-hand panel with RAG citations and router metadata |

## Repository layout

Folder layout matches Step 1 of the [project plan](./docs/project-plan.md#4-phased-milestone-plan). Implementation code lands in these folders as the build proceeds.

```text
.
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── docs/                                              # Design & planning docs
│   ├── README.md
│   ├── project-plan.md
│   └── system-design.md
├── app/                                               # FastAPI + LangGraph
├── ui/                                                # Streamlit dual-panel UI
├── rag/                                               # ChromaDB ingest & retrieval
├── data/                                              # services.json knowledge base
├── eval/                                              # eval.py + 30-case benchmark
└── tests/                                             # pytest (node + payload tests)
```

## Tech stack

**Core:** Python, FastAPI, Streamlit, LangGraph, ChromaDB, Pydantic, SQLite, Uvicorn, pytest.

**LLM:** `gpt-4o-mini` for intent classification and structured extraction (not for unconstrained therapy).

## Getting started

Environment setup is **Step 1**. Python **3.11–3.13** is recommended. After cloning:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in the environment (or a local `.env`) before running the Intent Gate.

Planned local run:

| Surface | URL |
|---------|-----|
| FastAPI | `http://localhost:8000` — `/chat/invoke` |
| Streamlit | `http://localhost:8501` |

See [Steps 5–7 and 13](./docs/project-plan.md#phase-1-architecture-data-schema-and-core-engine) for integration order.

## Build phases

Full step list: [project plan §4](./docs/project-plan.md#4-phased-milestone-plan).

| Phase | Steps | Focus |
|-------|-------|--------|
| 1 | 1–5 | Layout, `services.json`, GraphState, LangGraph, FastAPI + tests |
| 1 | 6–7 | Streamlit dual-panel UI |
| 2 | 8–10 | 30-case eval set + `eval.py` vs baseline |
| 2 | 11–14 | Latency / safety fallbacks, demo, dry run, deck freeze |

## If you need help now

If you or someone else is in immediate danger in Singapore, call **995** (SCDF) or **999** (police).

| Service | Contact | Notes |
|---------|---------|--------|
| [Samaritans of Singapore (SOS)](https://www.sos.org.sg/) | **1767** | 24/7 crisis / suicidal ideation |
| [Institute of Mental Health](https://www.imh.com.sg/) | **6389 2222** | Psychiatric emergency / helpline |
| [national mindline](https://mindline.sg) | **1771** | 24/7 mental health support ([MOH announcement](https://www.moh.gov.sg/newsroom/national-mindline-1771-to-provide--round-the-clock-support-for-mental-health/)) |
| [MOH mental health services](https://www.moh.gov.sg/seeking-healthcare/find-a-facility-or-service/mental-health-services/) | Directory | First-stop options including CREST and CHAT |

## References

| Document | Description |
|----------|-------------|
| [`docs/system-design.md`](./docs/system-design.md) | Runtime topology, API contracts, store split, Mermaid diagrams |
| [`docs/project-plan.md`](./docs/project-plan.md) | Phased plan, graph schema, risks, KPIs |
| [`docs/README.md`](./docs/README.md) | Docs index |
| [National Mental Health and Well-Being Strategy](https://go.gov.sg/mental-health-strategy) | Policy source for the Tiered Care Model |
| [MOH strategy launch](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) | Four-tier organisation of services |
| [MOHT — mindline.sg](https://www.moht.com.sg/our-programmes/mindline-sg/) | National digital first-stop (adjacent, not this product) |
| [Stepped care in Singapore (BMC)](https://link.springer.com/article/10.1186/s44263-025-00196-0) | Academic description of the Tiered Care Model |

---

*Non-clinical research / engineering prototype. Not a substitute for professional care.*
