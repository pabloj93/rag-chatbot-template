# Contributing

Thank you for your interest in contributing to `rag-chatbot-template`.

## How to contribute

1. **Fork** the repository and clone it locally.
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes — keep commits atomic and follow the existing code style.
4. **Test locally** with `docker compose up --build` and verify `/chat` works.
5. Run the eval harness and confirm recall does not regress: `python evaluation/run_eval.py`
6. Open a Pull Request with a clear description of what changed and why.

## What we welcome

- Bug fixes
- New doc sources in `backend/app/services/ingest.py` (`DOCS_URLS`)
- Retrieval quality improvements (reranking, hybrid search, chunking strategies)
- Frontend UX improvements (within the "minimal demo shell" constraint — see PRD)
- Evaluation improvements (new questions, better metrics, LLM-as-judge)
- Documentation and architecture clarity

## What to avoid

- Large new dependencies without clear benefit
- Breaking changes to the `/chat` or `/ingest` API contracts
- Committing `.env`, `data/raw/`, or any file with API keys

## Code style

- Python: small functions, explicit over clever, comments explain *why* not *what*
- TypeScript: same philosophy — minimal, readable, well-typed
- No premature abstractions — YAGNI

## Running tests

The project does not yet have unit tests (V3 goal). Use the eval harness as a functional test:

```bash
# 1. Start the backend
cd backend && uvicorn app.main:app --reload

# 2. Run ingest (first time only)
curl -X POST http://localhost:8000/ingest

# 3. Run evaluation
python evaluation/run_eval.py
```

A healthy run should show **>80% avg keyword recall** on the 15-question dataset.
