"""RAG generation chain.

Exposes three small building blocks that the chat router and the eval
harness compose into the full RAG pipeline:

    get_retriever()        -> VectorStoreRetriever for the Pinecone index.
    format_docs(docs)      -> Concatenated context string with [Source: url] tags.
    build_answer_chain()   -> LCEL chain that takes {context, question}
                              and returns the answer text. Supports both
                              .invoke() (eval) and .astream() (SSE).

Why three pieces instead of one big chain: the chat router needs the
retrieved docs *before* it streams (so it can emit a `sources` SSE event
first). Splitting the pipeline keeps the streaming flow obvious and avoids
running the retriever twice.
"""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings

logger = logging.getLogger(__name__)
from app.services.vectorstore import get_ensemble_retriever


# System prompt — "balanced" style (see PRD).
# Why this exact wording:
#   - First sentence anchors the assistant to the Anthropic docs domain.
#   - "Use the context as your primary source" is the *grounding* clause.
#   - "respond exactly: 'I don't have that...'" is the *refusal* clause —
#     prevents hallucinated answers when the docs don't cover the question.
#   - The "natural greetings" clause keeps UX humane — a "hi" must not be
#     refused.
#   - The "Sources:" footer is the *citation-forcing* clause.
SYSTEM_PROMPT = """You are an assistant for the Anthropic / Claude documentation.
Use the context below as your primary source. If the user's question requires
documentation content and the context does not contain the answer, respond
exactly: "I don't have that in the documentation." Do not guess. Do not draw
on general knowledge for technical questions.

For greetings, thanks, or meta-questions about your purpose, you may respond
naturally without citing sources.

When you answer with documentation content, end your message with a
"Sources:" section listing the URLs from the context that you actually used.

Context:
{context}
"""


def contextualize_query(question: str, history: list[BaseMessage]) -> str:
    """Rewrite `question` as a standalone retrieval query using conversation history.

    Why this exists: short follow-ups like "what's the cost?" are ambiguous
    without context. The retriever only sees the query string — it has no
    access to the conversation history. Rewriting to "what is the cost of
    prompt caching?" gives BM25 + Pinecone the terms they need to find the
    right chunks.

    The original `question` is still passed to the LLM chain so Claude answers
    in the user's natural voice. Only retrieval uses the rewritten query.

    Returns the original question unchanged if there is no history.
    Typical cost: ~100 tokens on Haiku = ~$0.00003 per call.
    """
    if not history:
        return question

    # Last 2 turns (4 messages) is enough context without burning extra tokens.
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:300]}"
        for m in history[-4:]
    )

    llm = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        temperature=0,
        max_tokens=80,
    )

    prompt = (
        "Rewrite the follow-up as a standalone search query. "
        "Return ONLY the rewritten query, no extra text. "
        "If it is already standalone, return it unchanged.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone query:"
    )

    result = llm.invoke(prompt)
    rewritten = result.content.strip().strip('"').strip("'")
    if rewritten and rewritten != question:
        logger.debug("Query contextualized: %r -> %r", question, rewritten)
    return rewritten or question


def get_retriever():
    """Return the hybrid BM25 + vector retriever (stage-1 of retrieve-then-rerank).

    Delegates to `get_ensemble_retriever()` in vectorstore.py, which:
      1. Runs BM25 (keyword) and vector (semantic) searches in parallel.
      2. Merges the two ranked lists via Reciprocal Rank Fusion (RRF).
      3. Returns the top `top_k_candidates` (default 20) merged docs.

    The chat router then calls `rerank()` (services/reranker.py) to narrow
    those 20 down to `top_k` (default 5) using the cross-encoder.

    Why three stages? Each adds a different signal:
      BM25     → exact keyword matches ("web_search", "web_fetch")
      Vector   → semantic / paraphrased matches
      Reranker → precision — reads (query, chunk) together
    """
    return get_ensemble_retriever()


def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved docs into the `{context}` string the prompt expects.

    Each chunk is prefixed with its source URL so the model can decide
    which URLs to list in its trailing "Sources:" section.
    """
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_answer_chain() -> Runnable:
    """Return the LCEL chain that produces the answer text.

    Input shape : {
        "context":  str,                    # formatted retrieved chunks
        "history":  list[BaseMessage],      # prior turns from the client
        "question": str,                    # the new user question
    }
    Output      : str (the answer, streamed token-by-token via .astream)

    Why this shape: the caller is in charge of running the retriever and
    formatting docs into `context`. The chain itself is just
    "prompt -> LLM -> parser", which makes streaming trivial — both
    .invoke() and .astream() work with no extra wiring.

    Why `history` lives in the input dict: conversation history is
    frontend-led (industry-standard for stateless backends — see PRD §1).
    The client sends prior turns with each request; the server stays
    stateless so it survives HF Spaces sleeps and multi-replica deploys.
    Pass an empty list on the first turn.
    """
    # Haiku in dev (cheap, fast); Sonnet for the final demo recording.
    # Switching is a one-line .env change (CLAUDE_MODEL=claude-sonnet-4-6).
    llm = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        # Low temperature for grounded factual answers — we want the model
        # to follow the context closely, not paraphrase creatively.
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            # MessagesPlaceholder expands to zero-or-more prior turns sent
            # by the client. On the first turn the caller passes [], and
            # the placeholder simply disappears from the rendered prompt.
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    # LCEL pipeline. StrOutputParser is the piece that makes .astream()
    # yield plain strings (token chunks) instead of AIMessageChunk objects.
    return prompt | llm | StrOutputParser()


def make_source(doc: Document) -> dict:
    """Convert a retrieved Document to the {url, snippet} shape the frontend uses.

    Why a small helper: both the chat router and the eval harness build
    sources from retrieved docs; centralizing the shape here avoids drift.
    """
    return {
        "url": doc.metadata.get("source", ""),
        # ~200 chars is enough for a 2-3 line preview in a <details>
        # block on the frontend, without bloating the JSON payload.
        "snippet": doc.page_content[:200].strip(),
    }
