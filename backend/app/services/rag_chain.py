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

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.services.vectorstore import get_vectorstore


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


def get_retriever() -> VectorStoreRetriever:
    """Return the Pinecone-backed retriever configured for top-k similarity.

    Why a separate function: the chat router needs the retrieved docs
    *before* it streams (to emit a `sources` SSE event first). Exposing
    the retriever here means the router calls it directly without
    re-running it inside the answer chain.
    """
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})


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
