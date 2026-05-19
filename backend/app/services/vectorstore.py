"""Pinecone vector store + BM25 hybrid retriever.

Exposes three factory functions:

    get_vectorstore()          → PineconeVectorStore for ingestion and
                                 pure-vector retrieval.

    get_ensemble_retriever()   → EnsembleRetriever (BM25 + vector, RRF
                                 fusion). Falls back to vector-only if
                                 the on-disk markdown cache doesn't exist.

    ensure_index()             → idempotent Pinecone index creation.

Why hybrid: BM25 catches exact keyword matches ("web_search", "web_fetch")
that cosine similarity misses; vector search catches semantic matches.
Reciprocal Rank Fusion (RRF) merges both signals — docs that rank high in
*both* retrievers float to the top.
"""

import hashlib
import logging
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


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

    _pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
    )


def get_vectorstore() -> PineconeVectorStore:
    """Return a LangChain VectorStore wrapping our Pinecone index."""
    ensure_index()
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )


# ---------------------------------------------------------------------------
# BM25 + Hybrid helpers
# ---------------------------------------------------------------------------

# Lazy-loaded BM25 index. Built once on first `get_ensemble_retriever` call.
# Why lazy: building BM25 requires re-chunking ~4800 docs from disk (~3 s).
# We don't want that cost at import time, only when the first query arrives.
_bm25_retriever: BM25Retriever | None = None


def _load_chunks_for_bm25() -> list[Document]:
    """Re-build chunks from the on-disk markdown cache for BM25 indexing.

    Uses the exact same splitter settings as `ingest.py` so BM25 and
    Pinecone cover the same chunk universe (same text → same sources).

    Why re-chunk from disk instead of fetching from Pinecone: Pinecone's
    SDK has no "fetch all vectors as text" API. Re-chunking the cache is
    fast (~3 s) and produces identical results.

    Returns an empty list if `data/raw/` is missing (pre-ingest state).
    """
    # Deferred import to avoid circular dependency:
    # ingest.py → get_vectorstore() (this file) → ingest.py would be circular
    # if imported at module level. Importing inside the function is safe.
    from app.services.ingest import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_URLS  # noqa: PLC0415
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

    # Map sha1(url) → url so we can recover the source URL from the filename.
    url_map = {hashlib.sha1(url.encode()).hexdigest(): url for url in DOCS_URLS}

    # data/raw/ lives 3 levels above this file:
    # backend/app/services/vectorstore.py → backend/data/raw/
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    if not cache_dir.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for md_file in cache_dir.glob("*.md"):
        url = url_map.get(md_file.stem, "")
        if not url:
            continue  # orphaned cache file from a URL no longer in DOCS_URLS
        text = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_documents(
            [Document(page_content=text, metadata={"source": url})]
        )
        all_chunks.extend(chunks)

    logger.info("BM25: loaded %d chunks from %s", len(all_chunks), cache_dir)
    return all_chunks


def get_ensemble_retriever() -> EnsembleRetriever | VectorStoreRetriever:
    """Return a hybrid BM25 + vector retriever using Reciprocal Rank Fusion.

    Both retrievers fetch `top_k_candidates` (default 20) docs each.
    RRF merges the two ranked lists: docs that appear high in both rank
    much higher than docs only one retriever found.

    BM25 weight 0.4 / vector weight 0.6 — tuned for technical docs where
    semantic understanding (vector) matters slightly more than exact keyword
    matching (BM25), but keyword precision is still important.

    Falls back to vector-only if BM25 can't be built (cache missing).
    """
    global _bm25_retriever

    vector_retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k_candidates}
    )

    if _bm25_retriever is None:
        chunks = _load_chunks_for_bm25()
        if not chunks:
            logger.warning(
                "data/raw/ not found — hybrid search unavailable. "
                "Run POST /ingest to populate the cache, then restart."
            )
            return vector_retriever
        _bm25_retriever = BM25Retriever.from_documents(
            chunks, k=settings.top_k_candidates
        )

    return EnsembleRetriever(
        retrievers=[_bm25_retriever, vector_retriever],
        # BM25 0.4 + vector 0.6 = 1.0. Increase BM25 weight if your queries
        # tend to contain exact technical terms; increase vector weight for
        # more conceptual / paraphrased queries.
        weights=[0.4, 0.6],
    )
