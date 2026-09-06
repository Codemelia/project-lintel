# `ui/` — Streamlit frontend

Dual-panel **service navigator chatbot**: `st.columns([2, 1])`. Left: chat. Right: intent confidence, active router node, RAG citations, `model_backend` / `egress` (**Trace Transparency** KPI).

This process talks to FastAPI over HTTP (`NAVIGATOR_API_URL`). It must not load `OPENAI_API_KEY`.

See [Steps 6–7](../docs/project-plan.md#steps-67--streamlit-dual-panel-interface), [trace protocol](../docs/system-design.md#8-trace-protocol-ui-binding), and [root README](../README.md). Planned URL: `http://localhost:8501`.
