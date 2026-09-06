"""Node 4 retrieval: metadata hard-filter, then vector similarity."""

from typing import Any

from .chroma_client import get_chroma_collection

HARD_STOP_SERVICE_IDS = frozenset({"sg-sos-01", "sg-imh-emergency-01"})

# Helper function for metadata filtering
def _where(
    *,
    age: int | None,
    cost_model: str | None,
) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [{"is_hard_stop_only": False}]
    if age is not None:
        clauses.append({"age_min": {"$lte": age}})
        clauses.append({"age_max": {"$gte": age}})
    if cost_model is not None:
        clauses.append({"cost_model": cost_model})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# Main retrieval function
# hard eligibility filtering is done in the retrieval step to avoid egress of sensitive user text
def retrieve_services(
    query: str,
    *,
    age: int | None = None,
    cost_model: str | None = None,
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """Return eligible catalogue hits. Never includes is_hard_stop_only rows."""
    
    # 1. Get collection and query
    collection = get_chroma_collection()
    raw = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=_where(age=age, cost_model=cost_model),
        include=["documents", "metadatas", "distances"],
    )
    
    # 2. Process query results
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    # 3. Filter hits
    hits: list[dict[str, Any]] = []
    for sid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        if sid in HARD_STOP_SERVICE_IDS or (meta or {}).get("is_hard_stop_only"):
            continue
        hits.append(
            {
                "service_id": sid,
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            }
        )
    return hits
