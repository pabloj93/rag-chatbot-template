"""LangFuse tracing setup.

Two responsibilities:

  1. Build a `CallbackHandler` per request. LangChain auto-emits the
     retriever, prompt, LLM and parser as nested spans into LangFuse
     through this handler — no manual instrumentation needed.

  2. Expose a way to enrich the resulting trace with per-request
     metadata (latency_ms, num_sources) *after* the chain finishes.

Why a fresh handler per request: a handler in LangFuse v2 represents one
trace. Reusing one across requests would collapse everything into a
single mega-trace. The underlying SDK client is a process-level
singleton, so this is cheap.
"""

from uuid import uuid4

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from app.config import settings


# Process-level LangFuse client. The CallbackHandler talks to LangFuse
# Cloud on its own, but we also need a client here to update traces
# *after* they're emitted (with latency_ms, num_sources, etc).
_langfuse_client: Langfuse | None = None


def _get_client() -> Langfuse:
    """Return the singleton LangFuse client, instantiating on first call."""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse_client


def make_session_id(provided: str | None = None) -> str:
    """Resolve the session_id for this request.

    Frontend-led: if the client passes one (e.g., via `X-Session-Id`
    header it set from localStorage), trust it; otherwise mint a fresh
    UUID4. Centralizing the rule here keeps every router consistent.
    """
    return provided or str(uuid4())


def make_langfuse_handler(session_id: str) -> CallbackHandler:
    """Build a LangFuse CallbackHandler for one request.

    Constant metadata (model, app_env) is attached here so it appears on
    every trace automatically. The router is still expected to call
    `enrich_trace` afterwards with values only known after the chain
    finishes (latency_ms, num_sources).
    """
    # Make sure the singleton client is alive — handy for `enrich_trace`
    # later, and harmless when the handler creates its own client.
    _get_client()
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id="anonymous",
        metadata={
            "model": settings.claude_model,
            "app_env": settings.app_env,
        },
    )


def enrich_trace(trace_id: str, **metadata) -> None:
    """Attach per-request metadata to an already-emitted trace.

    Called by the chat router after the chain returns, with values that
    the handler doesn't capture by default — typically `latency_ms` and
    `num_sources`. LangFuse upserts on trace id, so calling this with
    new keys merges instead of overwriting.
    """
    _get_client().trace(id=trace_id, metadata=metadata)
