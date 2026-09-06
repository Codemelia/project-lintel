import json
import sys
from pathlib import Path
from typing import Any

from .chroma_client import COLLECTION_NAME, get_chroma_collection

_REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICES_JSON_PATH = _REPO_ROOT / "data" / "services.json"

# Helper functions for metadata flattening
def _scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def flatten_metadata(service: dict[str, Any]) -> dict[str, Any]:
    """Map a schema record to Chroma-safe scalar metadata."""
    contact = service["contact"]
    return {
        "name": service["name"],
        "category": service["category"],
        "age_min": service["age_min"],
        "age_max": service["age_max"],
        "cost_model": service["cost_model"],
        "urgency_level": service["urgency_level"],
        "access_pathway": service["access_pathway"],
        "is_hard_stop_only": service["is_hard_stop_only"],
        "tier_labels_csv": ",".join(service["tier_labels"]),
        "eligibility_criteria": " | ".join(service["eligibility_criteria"]),
        "contact_website": _scalar(contact.get("website")),
        "contact_phone": _scalar(contact.get("phone")),
        "contact_operating_hours": _scalar(contact.get("operating_hours")),
        "location_type": contact["location_type"],
    }

# Embedding function
def embed_document(service: dict[str, Any]) -> str:
    return f"{service['name']}. {service['summary']} {service['access_pathway']}"

# Main ingestion function
def run_ingestion() -> int:
    
    # 1. Load catalogue
    if not SERVICES_JSON_PATH.is_file():
        print(f"Error: services.json not found at {SERVICES_JSON_PATH}", file=sys.stderr)
        return 1

    services: list[dict[str, Any]] = json.loads(SERVICES_JSON_PATH.read_text(encoding="utf-8"))
    if len(services) != 8:
        print(f"Error: expected 8 services, got {len(services)}", file=sys.stderr)
        return 1

    ids = [row["service_id"] for row in services]
    if len(set(ids)) != 8:
        print("Error: service_id values must be unique", file=sys.stderr)
        return 1

    # 2. Upsert documents into collection
    collection = get_chroma_collection()
    collection.upsert(
        ids=ids,
        documents=[embed_document(row) for row in services],
        metadatas=[flatten_metadata(row) for row in services],
    )

    # 3. Verify collection contents
    count = collection.count()
    if count != 8:
        print(f"Error: collection count is {count}, expected 8", file=sys.stderr)
        return 1

    stored = collection.get(ids=ids, include=["metadatas"])
    hard_stop_ids = {
        sid
        for sid, meta in zip(stored["ids"], stored["metadatas"] or [])
        if meta and meta.get("is_hard_stop_only")
    }
    expected_hard_stop = {"sg-sos-01", "sg-imh-emergency-01"}
    if hard_stop_ids != expected_hard_stop:
        print(
            f"Error: is_hard_stop_only ids {hard_stop_ids} != {expected_hard_stop}",
            file=sys.stderr,
        )
        return 1

    print(f"Upserted {len(services)} services into {COLLECTION_NAME} (count={count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_ingestion())
