# `rag/` — Retrieval and ChromaDB

Ingest [`data/services.json`](../data/services.json) into ChromaDB (`rag/ingest.py`): **upsert by `service_id`**, flatten nested `contact` to scalar metadata, persist under `CHROMA_PATH` (default: repo-root `chroma/`, gitignored). Path is resolved from the repo root, not the process cwd.

Retrieval (`rag/retrieve.py`) **must** apply metadata filters **before** vector similarity:

1. `age_min <= age <= age_max`
2. `cost_model` matches extracted cost
3. `is_hard_stop_only == false` (SOS / IMH never returned from Node 4)

Embeddings use a **local** sentence-transformers model (`EMBEDDING_MODEL`, default `all-MiniLM-L6-v2`) so catalogue vectors and Node 4 query embeddings are not sent to OpenAI. Intent and extract still use `gpt-4o-mini`.

See [Node 4](../docs/project-plan.md#node-4--metadata-hard-filter--rag), [Step 2](../docs/project-plan.md#step-2--knowledge-base-construction), and [system design §7](../docs/system-design.md#7-data-stores).
