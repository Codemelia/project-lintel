import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PATH", str(_REPO_ROOT / "chroma"))
COLLECTION_NAME = "sg_mh_services"

def get_chroma_collection():
    """
    Returns or creates a persistent ChromaDB collection with sentence-transformers embeddings.
    """
    
    # 1. Initialise persistent client
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR
    )
    
    # 2. Set up local embedding function (no API keys needed)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # 3. Get or create collection
    # Using the HNSW index for search by cosine distance, not L2 (Euclidean) distance.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )
    
    return collection