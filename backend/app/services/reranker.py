"""Cross-encoder reranker (two-stage retrieval — stage 2).

Stage 1: vector search fetches `top_k_candidates` (default 20) via cosine
similarity — fast, broad, sometimes imprecise.

Stage 2 (this file): cross-encoder scores every (query, chunk) pair and
keeps the best `top_k` (default 5) — slower but far more accurate because
the model reads query and document *together* instead of comparing
independent vectors.

Why a separate file: same singleton pattern as `embeddings.py`. The
cross-encoder takes ~3 s to load from disk; we load it once at first use
and reuse it for every query.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Trained on MS MARCO, a large-scale search QA dataset — matches our
    doc-retrieval use case directly.
  - ~80 MB, CPU-only, no API key. Supported by `sentence-transformers`
    which is already in requirements.txt.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings


# Module-level cache — same rationale as `_embeddings` in embeddings.py.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Return the singleton cross-encoder, loading from disk on first call."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Score every (query, doc) pair and return the top_n most relevant docs.

    Why cross-encoder beats cosine similarity here: it sees query + document
    together in a single forward pass, so it can pick up on exact term
    matches and fine-grained relevance cues that a bi-encoder misses.

    Example: "how many cache breakpoints are allowed?" → cosine retrieves
    chunks *about* cache breakpoints; cross-encoder finds the chunk that
    *states the limit* ("up to 4 explicit breakpoints").
    """
    if not docs:
        return docs

    model = get_reranker()
    # One forward pass per candidate — acceptable cost for 20 docs, not 4788.
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)

    # Sort descending by score and return the top_n.
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
