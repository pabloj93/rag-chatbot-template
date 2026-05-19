"""FastAPI app entry point.

Wires up logging, CORS, and the three routers (`/chat`, `/ingest`,
`/health`). Importing this module is the contract uvicorn uses:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import chat, ingest

# Logging is configured once, here, at import time. Each module then
# uses `logging.getLogger(__name__)` and inherits this config.
logging.basicConfig(level=settings.log_level)


app = FastAPI(
    title="RAG Chatbot — Anthropic Docs",
    description=(
        "A meta RAG chatbot that answers questions about the Anthropic / Claude "
        "documentation using Claude itself."
    ),
    version="0.1.0",
)


# CORS — the frontend runs on a different port in dev (Vite=5173,
# FastAPI=8000) and on a different subdomain on HF Spaces. We allow any
# origin here because the project is meant to be cloned and demo'd by
# strangers; tighten in production by replacing "*" with a list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.include_router(chat.router)
app.include_router(ingest.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe. Docker Compose and HF Spaces hit this on boot."""
    return {"status": "ok", "env": settings.app_env}


# Serve the React SPA when running inside the single-container HF Space
# (Dockerfile.hf copies the frontend build to /app/dist).
# In local dev, dist/ does not exist — Vite's dev server handles the
# frontend instead, so this block is silently skipped.
_DIST = Path(__file__).parent.parent / "dist"
if _DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_DIST / "assets")),
        name="static-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(_: str) -> FileResponse:
        """Catch-all: return index.html so React Router handles routing."""
        return FileResponse(str(_DIST / "index.html"))
