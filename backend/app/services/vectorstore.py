"""Pinecone vector store.

Exposes one factory function (`get_vectorstore`) that other modules use
to add documents and to build retrievers for the RAG chain. The Pinecone
index is created on first use if it does not already exist, so the app is
plug-and-play from a fresh clone.
"""

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.services.embeddings import get_embeddings


# Single Pinecone client for the whole process.
# Why module-level: the client is just an authenticated HTTP wrapper, safe
# to share and re-using it avoids re-doing the auth handshake on every call.
_pc = Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    """Create the Pinecone index if it does not exist yet.

    Why we auto-create: keeps the project plug-and-play. A reviewer can
    `git clone` and run without ever opening the Pinecone dashboard.
    The first boot pays ~5s for the create call; later boots are no-ops.
    """
    existing = [i.name for i in _pc.list_indexes()]
    if settings.pinecone_index_name in existing:
        return

    # Serverless free tier on AWS us-east-1 covers our entire footprint
    # (~5k vectors, 384-dim) well under the 100k-vector quota.
    # Metric is `cosine` because our embeddings are L2-normalized
    # (see embeddings.py — normalize_embeddings=True).
    _pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
    )


def get_vectorstore() -> PineconeVectorStore:
    """Return a LangChain VectorStore wrapping our Pinecone index.

    Why this wrapper instead of the raw Pinecone client: the LangChain
    VectorStore exposes `.add_documents(...)` for ingestion and
    `.as_retriever(...)` for the RAG chain, both with stable interfaces
    that survive vector-DB swaps later (e.g., Chroma, Weaviate).
    """
    ensure_index()
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )
