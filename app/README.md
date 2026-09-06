# `app/` — FastAPI backend and LangGraph engine

Home for the REST API (`/chat/invoke`), `GraphState`, and LangGraph nodes (Intent Gate → extraction → tier map → RAG → decision / safety fallback).

See [system design](../docs/system-design.md) (API + graph compile) and [project plan §2–3](../docs/project-plan.md#2-system-architecture-and-technical-stack). Product context: [root README](../README.md#architecture).
