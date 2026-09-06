import json
from pathlib import Path

from rag.ingest import flatten_metadata

_SERVICES = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "services.json").read_text(
        encoding="utf-8"
    )
)


def test_flatten_contact_and_lists() -> None:
    sos = next(row for row in _SERVICES if row["service_id"] == "sg-sos-01")
    meta = flatten_metadata(sos)
    assert meta["contact_phone"] == "1767"
    assert meta["contact_website"] == "https://www.sos.org.sg"
    assert "contact" not in meta
    assert meta["tier_labels_csv"] == "4"
    assert meta["is_hard_stop_only"] is True
    assert "Call 1767" in meta["access_pathway"]


def test_flatten_null_phone() -> None:
    private = next(
        row for row in _SERVICES if row["service_id"] == "sg-private-counseling-01"
    )
    meta = flatten_metadata(private)
    assert meta["contact_phone"] == ""
    assert meta["is_hard_stop_only"] is False
