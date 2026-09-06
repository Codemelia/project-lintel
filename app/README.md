# `app/` — FastAPI backend and LangGraph engine

Home for the REST API (`/chat/invoke`), `GraphState`, LangGraph nodes (Intent Gate → extraction → tier map → RAG → decision / safety fallback), and the OpenAI structured-output client for Nodes 1–2.

Planned layout (see [system design §3](../docs/system-design.md#3-component-responsibilities)):

```text
app/
  __init__.py
  safety.py          # regex + FALLBACK_COPY (SOS 1767, IMH 6389 2222)
  state.py           # GraphState TypedDict / Pydantic — Step 3
  graph/models.py    # get_node_model(); OpenAI JSON for intent + extract
  graph.py           # StateGraph compile — Step 4
  nodes/             # one module per node; no FastAPI imports — Step 4
  schemas.py         # ChatRequest / ChatResponse / TraceEvent — Step 5
  main.py            # FastAPI app — Step 5
  timeouts.py        # asyncio.wait_for around model calls
```

**`GraphState` lives in `app/state.py`.** Field list: [project plan §3.1](../docs/project-plan.md#31-graphstate-pydantic--typeddict). Import as `from app.state import GraphState`.

Present today: `__init__.py`, `safety.py` (first-person regex + `FALLBACK_COPY`), `state.py` (`GraphState`), `graph/models.py` (`get_node_model`, `classify_intent`, `extract_parameters`). `graph.py` / `nodes/` / FastAPI land in Steps 4–5.

`classify_intent` is regex first; a match never calls OpenAI (`egress=false`). All other Node 1–2 calls send the utterance to `gpt-4o-mini` (`egress=true`). Model `confidence` is logged on the trace only; it does not choose the next node and is not a crisis cutoff.

Streamlit must not import this package for inference. Product context: [root README](../README.md#architecture).
