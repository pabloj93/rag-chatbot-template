"""Embedding model loader.

Wraps `sentence-transformers/all-MiniLM-L6-v2` via LangChain's
HuggingFaceEmbeddings adapter. The model is loaded lazily on first call
and reused for the lifetime of the process.

Why we wrap via LangChain instead of using sentence-transformers directly:
PineconeVectorStore expects a LangChain `Embeddings` interface, and
swapping to a different embedding model later becomes a one-line change.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings


# Module-level cache for the singleton instance.
# Why: Python imports each module exactly once, so this variable is
# naturally shared by every caller — no thread-safety dance needed.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the singleton embeddings model, loading it on first call.

    Why singleton: loading the model from disk takes ~3 seconds and ~80MB
    of RAM. Reloading per request would make every /chat call painfully
    slow and quickly run the container out of memory.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            # HF Spaces free tier is CPU-only — pinning the device here
            # avoids surprises when the same code runs on a GPU box.
            model_kwargs={"device": "cpu"},
            # Cosine similarity (Pinecone default) is equivalent to a dot
            # product on unit-length vectors. Normalizing here lets the
            # downstream math be cheaper without changing the result.
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
