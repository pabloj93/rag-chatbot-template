"""POST /ingest router.

Triggers a full re-scrape and re-upload of the docs index. This is a
heavy operation (~3 min cold, ~30s warm from the on-disk HTML cache) and
in V1 we run it once at first boot via Swagger UI to keep the project
plug-and-play.
"""

from fastapi import APIRouter

from app.services.ingest import ingest_documents

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("")
def run_ingest() -> dict:
    """Scrape configured URLs, split into chunks, upsert to Pinecone.

    Idempotent: each chunk is stored under a stable ID derived from its
    URL + chunk-index, so re-running upserts the same vectors instead of
    duplicating them. Skips 404s silently and logs them.
    """
    return ingest_documents()
