"""Application configuration.

Loads values from `.env` (or real environment variables) and exposes them
as a typed `settings` singleton. Importing this module is enough to
guarantee that every required key exists — pydantic raises ValidationError
at import time if a required field is missing.

Why this file exists: single source of truth for env vars, fail-fast on
missing keys, and free type coercion ("5" -> 5, "true" -> True, etc).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Absolute path to the project-root .env file.
# Why absolute (not "./.env"): uvicorn is typically launched from `backend/`,
# while `.env` lives at the repo root. A relative path would silently miss
# the file. This computation walks up from `backend/app/config.py` to
# the project root.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Typed wrapper over the .env file."""

    # Tells pydantic where to read variables from and how to behave.
    # Why `extra="ignore"`: .env also holds frontend-only vars (VITE_*) —
    # we don't want pydantic to crash because of fields the backend ignores.
    # Why we still keep `env_file=...` when Docker injects env vars directly:
    # pydantic reads os.environ first, then falls back to the file, so this
    # path is harmless in containers and required for local `uvicorn` runs.
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    # --- Anthropic (Claude) ---
    # Why no default: a missing key must fail loudly at boot, not in the
    # middle of the first user request.
    anthropic_api_key: str
    # Haiku in dev (cheap, fast); Sonnet for the final demo and eval runs.
    claude_model: str = "claude-haiku-4-5-20251001"

    # --- Pinecone (vector DB) ---
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "anthropic-docs"

    # --- LangFuse (observability) ---
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Retrieval tuning ---
    # Two-stage retrieve-then-rerank pipeline:
    #   1. top_k_candidates: how many chunks the vector search fetches first
    #      (broad, fast cosine similarity).
    #   2. top_k: how many of those candidates the cross-encoder keeps
    #      (precise, slow — but only runs on top_k_candidates docs).
    top_k_candidates: int = 20
    top_k: int = 5
    # MiniLM is the free/local embedding baseline; 384-dim, fast on CPU.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Cross-encoder for reranking. Trained on MS MARCO (search QA) — same
    # use-case as our doc retrieval. CPU-only, ~80 MB, no API key needed.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"


# Instantiate once at import time.
# Why: pydantic validates here. If any required env var is missing,
# Python raises ValidationError before uvicorn ever binds the port.
# This is the "fail-fast" pattern — cheaper than crashing mid-request.
settings = Settings()
