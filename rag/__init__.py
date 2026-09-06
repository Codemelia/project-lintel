"""Chroma ingest and retrieval for the curated service catalogue."""

from .chroma_client import COLLECTION_NAME, get_chroma_collection

__all__ = ["COLLECTION_NAME", "get_chroma_collection"]
