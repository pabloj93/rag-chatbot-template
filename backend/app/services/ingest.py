"""Document ingestion pipeline.

Scrapes Anthropic / Claude documentation pages, cleans the HTML, splits
the text into ~800-char overlapping chunks, and upserts them to Pinecone
through the LangChain VectorStore wrapper.

The pipeline is idempotent: each chunk gets a stable ID derived from its
URL + chunk-index, so re-running `/ingest` overwrites rather than
duplicating vectors. Raw HTML is cached under `data/raw/` so repeated
dev runs do not hammer the Anthropic site.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


# On-disk cache for raw HTML between ingest runs.
# Why: avoids hitting platform.claude.com 99 times every time we tweak the
# splitter or rerun for debugging. The folder is in .gitignore.
CACHE_DIR = Path("data/raw")

# Splitter config — see PRD §3 (V1). 800/120 is the validated default for
# technical docs: large enough for context, small enough to keep retrieval
# precise. Separators are tried in order; keeps paragraph/line boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# All URLs we index. Grouped by theme for auditability.
# Total: 99 pages on platform.claude.com/docs/en/* — 74 from the sitemap
# plus 25 from sections (API ref, models, prompt-engineering, eval) that
# exist on the live site but were missing from the EN sitemap as of
# May 2026. URLs that 404 are skipped at fetch time, never crash ingest.
DOCS_URLS: list[str] = [
    # --- 1. Core & Getting Started ---
    "https://platform.claude.com/docs/en/intro",
    "https://platform.claude.com/docs/en/get-started",
    # --- 2. Messages API & Stop Reasons ---
    "https://platform.claude.com/docs/en/build-with-claude/overview",
    "https://platform.claude.com/docs/en/build-with-claude/working-with-messages",
    "https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons",
    # --- 3. Thinking & Effort Controls ---
    "https://platform.claude.com/docs/en/build-with-claude/extended-thinking",
    "https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking",
    "https://platform.claude.com/docs/en/build-with-claude/effort",
    "https://platform.claude.com/docs/en/build-with-claude/task-budgets",
    "https://platform.claude.com/docs/en/build-with-claude/fast-mode",
    # --- 4. Generation Modes ---
    "https://platform.claude.com/docs/en/build-with-claude/structured-outputs",
    "https://platform.claude.com/docs/en/build-with-claude/citations",
    "https://platform.claude.com/docs/en/build-with-claude/streaming",
    "https://platform.claude.com/docs/en/build-with-claude/batch-processing",
    "https://platform.claude.com/docs/en/build-with-claude/search-results",
    # --- 5. Context, Caching & Tokens ---
    "https://platform.claude.com/docs/en/build-with-claude/context-windows",
    "https://platform.claude.com/docs/en/build-with-claude/compaction",
    "https://platform.claude.com/docs/en/build-with-claude/context-editing",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
    "https://platform.claude.com/docs/en/build-with-claude/token-counting",
    # --- 6. Multimodal ---
    "https://platform.claude.com/docs/en/build-with-claude/files",
    "https://platform.claude.com/docs/en/build-with-claude/pdf-support",
    "https://platform.claude.com/docs/en/build-with-claude/vision",
    # --- 7. Embeddings & Multilingual ---
    "https://platform.claude.com/docs/en/build-with-claude/embeddings",
    "https://platform.claude.com/docs/en/build-with-claude/multilingual-support",
    # --- 8. Tool Use — Fundamentals ---
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls",
    # --- 9. Tool Use — Advanced Patterns ---
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming",
    # --- 10. Built-in Tools ---
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool",
    # --- 11. Tool Use — Reference & Troubleshooting ---
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-combinations",
    "https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool",
    # --- 12. Agent Skills ---
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices",
    "https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise",
    "https://platform.claude.com/docs/en/build-with-claude/skills-guide",
    # --- 13. MCP — Model Context Protocol ---
    "https://platform.claude.com/docs/en/agents-and-tools/remote-mcp-servers",
    "https://platform.claude.com/docs/en/agents-and-tools/mcp-connector",
    # --- 14. Managed Agents — Setup & Core ---
    "https://platform.claude.com/docs/en/managed-agents/overview",
    "https://platform.claude.com/docs/en/managed-agents/quickstart",
    "https://platform.claude.com/docs/en/managed-agents/onboarding",
    "https://platform.claude.com/docs/en/managed-agents/agent-setup",
    "https://platform.claude.com/docs/en/managed-agents/tools",
    "https://platform.claude.com/docs/en/managed-agents/mcp-connector",
    "https://platform.claude.com/docs/en/managed-agents/permission-policies",
    "https://platform.claude.com/docs/en/managed-agents/skills",
    # --- 15. Managed Agents — Runtime & Patterns ---
    "https://platform.claude.com/docs/en/managed-agents/environments",
    "https://platform.claude.com/docs/en/managed-agents/cloud-containers",
    "https://platform.claude.com/docs/en/managed-agents/sessions",
    "https://platform.claude.com/docs/en/managed-agents/events-and-streaming",
    "https://platform.claude.com/docs/en/managed-agents/webhooks",
    "https://platform.claude.com/docs/en/managed-agents/define-outcomes",
    "https://platform.claude.com/docs/en/managed-agents/dreams",
    "https://platform.claude.com/docs/en/managed-agents/multi-agent",
    # --- 16. Admin API ---
    "https://platform.claude.com/docs/en/manage-claude/admin-api",
    "https://platform.claude.com/docs/en/manage-claude/usage-cost-api",
    # --- 17. API Reference (not in EN sitemap but pages exist on the live site) ---
    "https://platform.claude.com/docs/en/api/overview",
    "https://platform.claude.com/docs/en/api/client-sdks",
    "https://platform.claude.com/docs/en/api/sdks/python",
    "https://platform.claude.com/docs/en/api/sdks/typescript",
    "https://platform.claude.com/docs/en/api/sdks/java",
    "https://platform.claude.com/docs/en/api/sdks/go",
    "https://platform.claude.com/docs/en/api/beta-headers",
    "https://platform.claude.com/docs/en/api/errors",
    "https://platform.claude.com/docs/en/api/rate-limits",
    "https://platform.claude.com/docs/en/api/versioning",
    "https://platform.claude.com/docs/en/api/service-tiers",
    # --- 18. About Claude — Models & Pricing ---
    "https://platform.claude.com/docs/en/about-claude/models/overview",
    "https://platform.claude.com/docs/en/about-claude/models/choosing-a-model",
    "https://platform.claude.com/docs/en/about-claude/models/migration-guide",
    "https://platform.claude.com/docs/en/about-claude/model-deprecations",
    "https://platform.claude.com/docs/en/about-claude/pricing",
    "https://platform.claude.com/docs/en/about-claude/glossary",
    # --- 19. Prompt Engineering ---
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-tools",
    # --- 20. Test & Evaluate ---
    "https://platform.claude.com/docs/en/test-and-evaluate/develop-tests",
    "https://platform.claude.com/docs/en/test-and-evaluate/eval-tool",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency",
    "https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks",
]


def _cache_path_for(url: str) -> Path:
    """Return the on-disk cache path for `url`, using a hashed filename.

    Why a hash: URLs contain `/` and other characters that aren't valid in
    filenames. We don't need to recover the URL from the path — the source
    URL is stored in each Document's metadata.
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.md"


def _fetch_markdown(url: str) -> str | None:
    """Fetch the markdown version of a docs page.

    Anthropic exposes a `.md` mirror of every page at `<url>.md` that
    returns clean markdown (no JS, no nav, no footer). This is *much*
    better than scraping the Next.js HTML, where article content is
    client-side rendered and BeautifulSoup only sees "Loading..."
    placeholders.

    Returns the markdown string, or `None` if the page 404'd or errored.
    """
    cache = _cache_path_for(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    md_url = url + ".md"
    try:
        resp = requests.get(
            md_url,
            timeout=30,
            headers={"User-Agent": "rag-chatbot-template/0.1"},
        )
    except requests.RequestException as e:
        logger.warning("Fetch failed for %s: %s", md_url, e)
        return None

    if resp.status_code == 404:
        # Some speculative URLs may legitimately not exist — skip silently
        # rather than crash the entire ingest.
        logger.info("Skipping 404: %s", md_url)
        return None
    resp.raise_for_status()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.text


def _build_documents() -> list[Document]:
    """Fetch every configured URL and wrap each markdown blob as a Document.

    The original (non-`.md`) URL is stored in metadata so the frontend can
    show a clickable link the user actually expects to see.
    """
    docs: list[Document] = []
    for url in DOCS_URLS:
        text = _fetch_markdown(url)
        if not text or not text.strip():
            logger.warning("Empty or missing content for %s", url)
            continue
        docs.append(Document(page_content=text, metadata={"source": url}))
    logger.info("Built %d documents from %d URLs", len(docs), len(DOCS_URLS))
    return docs


def _make_chunk_id(url: str, idx: int) -> str:
    """Return a stable Pinecone vector ID for a (url, chunk-index) pair.

    Why stable: re-running /ingest should overwrite vectors in place, not
    create duplicates. Passing explicit IDs to Pinecone upsert turns the
    operation into an idempotent "set this key to this value".
    """
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{url_hash}_{idx}"


def ingest_documents() -> dict:
    """Top-level pipeline: scrape -> clean -> split -> upsert.

    Returns a small stats dict the API exposes to the caller so a reviewer
    knows the ingest actually moved data into Pinecone.
    """
    raw_docs = _build_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separators are tried in order: paragraph -> line -> sentence ->
        # word -> char. Respecting natural boundaries keeps chunks readable.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Split per-URL so we can pair each chunk with a stable index.
    all_chunks: list[Document] = []
    all_ids: list[str] = []
    for doc in raw_docs:
        url = doc.metadata["source"]
        chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(_make_chunk_id(url, idx))

    if not all_chunks:
        return {"pages_fetched": len(raw_docs), "chunks": 0, "uploaded": False}

    vs = get_vectorstore()
    vs.add_documents(all_chunks, ids=all_ids)
    logger.info("Uploaded %d chunks to Pinecone", len(all_chunks))
    return {
        "pages_fetched": len(raw_docs),
        "chunks": len(all_chunks),
        "uploaded": True,
    }
