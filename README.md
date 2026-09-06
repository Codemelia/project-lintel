# Project Lintel: A Mental Health Service Navigator

A **public-facing, non-clinical service navigator chatbot** for Singapore’s mental health ecosystem. Help-seekers describe their situation in free text; a deterministic [LangGraph](https://langchain-ai.github.io/langgraph/) engine maps constraints to the [National Tiered Care Model](https://www.moh.gov.sg/newsroom/launch-of-national-mental-health-and-well-being-strategy/) (Tiers 1–4) and returns a traceable provider pathway—not therapy, diagnosis, or CBT.

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

The Navigator is a **chatbot that routes, not treats**: a transparent preparation assistant and service router behind a chat UI.

1. **Intent & Scope Gate** — first-person crisis regex, then `gpt-4o-mini` structured JSON (`in_scope` / `crisis` / `out_of_scope`). Crisis or out-of-scope requests (diagnosis, CBT therapy) never enter conversational generation.
2. **Parameter extraction** — age, `cost_model` (`Free` / `Subsidized` / `Private` / `Variable`), `urgency_level`, and `category` from free text (same OpenAI client).
3. **Tier mapping** — rule-based assignment to Tiers 1–4.
4. **Hard-filtered RAG** — metadata filters on curated Singapore providers **before** vector search (`age_min`/`age_max`, `cost_model`, `is_hard_stop_only=false`). Embeddings stay on MiniLM locally.
5. **Explainable recommendation** — templated pathway plus a live decision trace (intent, active node, citations, `model_backend` / `egress`).

Strategic focus is borrowed from front-door e-triage (for example UK NHS evaluations of Limbic Access): reduce clinician assessment burden by allocating pathways, not by acting as a therapist. Liability is bounded by refusing unconstrained therapeutic bots.

Measurable engineering KPIs: **Scope Adherence Rate**, **Service Routing Accuracy**, **Safety Hand-off Success**, and **Trace Transparency**. Full definitions are in the [evaluation section of the project plan](./docs/project-plan.md#5-evaluation-strategy-and-kpis).

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

Modular Python stack. Chat UI is Streamlit; routing and models live in FastAPI. **Diagrams, HTTP contracts, store split, and control-flow predicates:** [`docs/system-design.md`](./docs/system-design.md).

| Layer | Technology | Role |
|-------|------------|------|
| Frontend | [Streamlit](https://streamlit.io/) | Dual column: chatbot (left) and live reasoning trace (right) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | REST orchestration, [Pydantic](https://docs.pydantic.dev/) validation |
| State engine | [LangGraph](https://langchain-ai.github.io/langgraph/) + `app/graph/models.py` | Deterministic graph; models fill fields, Python chooses edges |
| Intent / extract | `gpt-4o-mini` structured JSON (`OPENAI_BASE_URL` optional) | Slot-filling only; not unconstrained therapy |
| Storage / RAG | SQLite + [ChromaDB](https://www.trychroma.com/) + MiniLM | Local catalogue vectors; curated provider metadata |

**Graph nodes** (predicates and sequence diagrams in [system design](./docs/system-design.md); node roles in the [project plan](./docs/project-plan.md#3-langgraph-engine-architecture)):

1. Intent & Safety Gate (regex → `gpt-4o-mini` if no regex match)
2. Parameter Extraction (`get_node_model("extract")`)
3. National Tier Mapping (rules, no generative model)
4. Metadata hard-filter + RAG
5. Decision & Explanation (templated pathway)
6. Safety Fallback & Boundary (hardcoded emergency contacts)

Conditional edges prevent unconstrained LLM loops. Crisis / out-of-scope paths skip generation entirely.

## Knowledge base

Eight primary providers spanning Tiers 1–4 in [`data/services.json`](./data/services.json) ([schema](./data/schemas/services.schema.json)). Ingest into ChromaDB via [`rag/`](./rag/README.md) (upsert by `service_id`; flatten nested `contact`; exclude `is_hard_stop_only` from Node 4).

| `service_id` | Tiers | Provider | Routing |
|--------------|-------|----------|---------|
| `sg-sos-01` | 4 | SOS (1767) | `is_crisis` → hard stop |
| `sg-imh-emergency-01` | 4 | IMH Emergency (`6389 2222`) | `is_crisis` → hard stop |
| `sg-mindline-01` | 1 | National Mindline (1771) | Free / 24/7 listening ear |
| `sg-chat-01` | 1–2 | CHAT | Age 16–30 / free assessment |
| `sg-polyclinic-telepsych-01` | 2 | Polyclinic tele-psychology | Subsidized / GP referral |
| `sg-fsc-01` | 2 | Family Service Centres | Subsidized / community casework |
| `sg-crest-01` | 1–2 | CREST | Age 18+ / outreach |
| `sg-private-counseling-01` | 3 | Private counselling | `Private` / rapid booking |

## Evaluation targets

Benchmark: **30** synthetic queries (15 in-scope navigation, 8 acute crisis, 7 out-of-scope). Compare **two** configurations: unconstrained `gpt-4o-mini` (baseline) vs Navigator chatbot. Pipeline: [`eval/`](./eval/README.md).

| Metric | Target | Method |
|--------|--------|--------|
| Safety Hand-off Success | **100%** | Crisis cases must hard-stop to emergency contacts; no conversational generation |
| Scope Adherence Rate | **> 90%** | Out-of-scope (CBT / diagnosis) refused and redirected to non-clinical navigation |
| Service Routing Accuracy | **> 85%** | Age, `cost_model`, and `urgency_level` match the eligible `service_id` in `services.json` |
| Trace Transparency | **100%** | Every output fills the right-hand panel with RAG citations and router metadata |

## Repository layout

Folder layout matches Step 1 of the [project plan](./docs/project-plan.md#4-phased-milestone-plan). Implementation code lands in these folders as the build proceeds.

```text
.
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── .env.example                                       # Env template (copy to .env)
├── docs/                                              # Design & planning docs
│   ├── README.md
│   ├── project-plan.md
│   └── system-design.md
├── app/                                               # FastAPI + LangGraph + OpenAI client
├── ui/                                                # Streamlit dual-panel chatbot
├── rag/                                               # ChromaDB ingest & retrieval
├── data/                                              # services.json + schemas/
├── eval/                                              # eval.py + 30-case two-config benchmark
└── tests/                                             # pytest (node + payload tests)
```

## Tech stack

**Core:** Python, FastAPI, Streamlit, LangGraph, ChromaDB, Pydantic, SQLite, Uvicorn, pytest.

**Models:** `gpt-4o-mini` for intent and extraction (JSON only). Catalogue embeddings: local `all-MiniLM-L6-v2`. Node 5–6 never generate therapy.

## Getting started

Environment setup is **Step 1**. Python **3.11–3.13** is recommended. After cloning:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

Copy [`.env.example`](./.env.example) to `.env` at the **repo root** (gitignored). Set `OPENAI_API_KEY`. Optional `OPENAI_BASE_URL` for a private/regional OpenAI-compatible endpoint. Streamlit needs `NAVIGATOR_API_URL` only.

Planned local run (public hosting is later; same API contract):

| Surface | URL |
|---------|-----|
| FastAPI | `http://localhost:8000` — `/chat/invoke` |
| Streamlit | `http://localhost:8501` |

See [Steps 5–7 and 13](./docs/project-plan.md#phase-1-architecture-data-schema-and-core-engine) for integration order.

## Build phases

Full step list: [project plan §4](./docs/project-plan.md#4-phased-milestone-plan).

| Phase | Steps | Focus |
|-------|-------|--------|
| 1 | 1–5 | Layout, `services.json`, GraphState, OpenAI client, LangGraph, FastAPI + tests |
| 1 | 6–7 | Streamlit dual-panel chatbot |
| 2 | 8–10 | 30-case eval set + two-config `eval.py` |
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
