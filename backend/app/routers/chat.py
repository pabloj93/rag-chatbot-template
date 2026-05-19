"""POST /chat (sync) and POST /chat/stream (SSE).

The non-streaming `/chat` endpoint exists for the eval harness and for
tools that can't speak SSE (Swagger UI, simple curl). The streaming
endpoint is what the frontend uses.

SSE event schema (4 event types):

    event: sources  -> [{url, snippet}, ...]      emitted once, before tokens
    event: token    -> {text: "..."}              emitted N times
    event: done     -> {trace_id, latency_ms}     emitted once at the end
    event: error    -> {message: "..."}           emitted at most once

Both endpoints share the same request schema and the same service layer
(rag_chain + tracing). Only the response shape differs.
"""

import json
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.services.rag_chain import (
    build_answer_chain,
    format_docs,
    get_retriever,
    make_source,
)
from app.services.reranker import rerank
from app.services.tracing import (
    enrich_trace,
    make_langfuse_handler,
    make_session_id,
)

router = APIRouter(prefix="/chat", tags=["chat"])


# --- Request / response schemas -----------------------------------------

class Message(BaseModel):
    """One turn in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Body shared by `/chat` and `/chat/stream`.

    Why a unified `messages` list (instead of separate `question` and
    `history` fields): matches the shape of the Anthropic Messages API,
    so a developer who already knows that API can plug in directly.
    The last message must be the new user question.
    """

    messages: list[Message] = Field(..., min_length=1)
    session_id: str | None = None


# --- Helpers ------------------------------------------------------------

def _split_messages(messages: list[Message]) -> tuple[list[BaseMessage], str]:
    """Split the client-supplied list into `(history, current_question)`.

    The last message must be from the user (it's the new question).
    Everything before it becomes the history we feed to MessagesPlaceholder.
    """
    if messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="Last message must have role='user'."
        )
    history: list[BaseMessage] = []
    for m in messages[:-1]:
        # Map our wire-format roles to LangChain message classes.
        cls = HumanMessage if m.role == "user" else AIMessage
        history.append(cls(content=m.content))
    return history, messages[-1].content


def _sse(event: str, data) -> str:
    """Format a payload as a single SSE event frame.

    SSE spec: each frame is `event: <name>\\ndata: <json>\\n\\n`. The
    double newline is what tells the browser the event is complete.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- Endpoints ----------------------------------------------------------

@router.post("")
def chat(req: ChatRequest) -> dict:
    """Non-streaming chat: full request/response in a single round-trip.

    Used by the eval harness and by anything that can't consume SSE.
    Returns the answer plus the sources and the LangFuse trace_id so
    callers can correlate logs.
    """
    history, question = _split_messages(req.messages)
    session_id = make_session_id(req.session_id)
    handler = make_langfuse_handler(session_id)

    t0 = time.perf_counter()
    # Stage 1: vector search returns top_k_candidates (broad).
    # Stage 2: cross-encoder re-ranks to top_k (precise).
    docs = rerank(question, get_retriever().invoke(question), top_n=settings.top_k)
    answer = build_answer_chain().invoke(
        {
            "context": format_docs(docs),
            "history": history,
            "question": question,
        },
        config={"callbacks": [handler]},
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    sources = [make_source(d) for d in docs]

    trace_id = handler.get_trace_id()
    enrich_trace(trace_id, latency_ms=latency_ms, num_sources=len(sources))
    # Flush the handler's internal client before returning so the trace
    # actually lands in LangFuse Cloud — the default batcher only flushes
    # every ~few seconds, and in a CLI smoke test the request often ends
    # first, dropping the trace on the floor.
    handler.flush()

    return {
        "answer": answer,
        "sources": sources,
        "trace_id": trace_id,
        "latency_ms": latency_ms,
        "session_id": session_id,
    }


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat via SSE. Emits `sources` -> N x `token` -> `done`.

    The retriever runs first (sync, ~100ms) so we can emit the `sources`
    event before the LLM begins generating. Only the LLM generation is
    truly streamed token-by-token.
    """
    history, question = _split_messages(req.messages)
    session_id = make_session_id(req.session_id)
    handler = make_langfuse_handler(session_id)

    async def event_stream():
        t0 = time.perf_counter()
        try:
            # Stage 1: vector search (broad). Stage 2: cross-encoder (precise).
            # Sources emitted first so the frontend renders them while tokens arrive.
            docs = rerank(question, get_retriever().invoke(question), top_n=settings.top_k)
            sources = [make_source(d) for d in docs]
            yield _sse("sources", sources)

            # 2. Stream the LLM answer.
            chain = build_answer_chain()
            async for token in chain.astream(
                {
                    "context": format_docs(docs),
                    "history": history,
                    "question": question,
                },
                config={"callbacks": [handler]},
            ):
                yield _sse("token", {"text": token})

            # 3. Final event so the frontend can stop the spinner and
            #    log the trace id for debugging.
            latency_ms = int((time.perf_counter() - t0) * 1000)
            trace_id = handler.get_trace_id()
            enrich_trace(trace_id, latency_ms=latency_ms, num_sources=len(sources))
            yield _sse(
                "done",
                {
                    "trace_id": trace_id,
                    "latency_ms": latency_ms,
                    "session_id": session_id,
                },
            )
        except Exception as e:
            # Never raise mid-stream: the client would just see a dropped
            # TCP connection with no signal. Emit a structured error event
            # so the frontend can show a meaningful message.
            yield _sse("error", {"message": str(e)})
        finally:
            # Same reason as the sync endpoint — flush before the request
            # closes so the trace lands in LangFuse Cloud.
            handler.flush()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # These three headers are critical: they tell every proxy on the
        # path (nginx, HF Spaces) NOT to buffer the response. Without
        # them, SSE feels broken — the client sees nothing for ~5s, then
        # receives every event in one burst.
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
