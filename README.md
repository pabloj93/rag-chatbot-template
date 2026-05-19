# rag-chatbot-template

> A production-grade RAG chatbot that answers questions about the **Anthropic / Claude documentation** — built with Claude itself. A meta portfolio piece demonstrating end-to-end mastery of the modern LLM engineering stack.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C.svg)](https://python.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What this demonstrates

- **RAG architecture end-to-end** — chunking, embedding, retrieval, grounding, citation enforcement
- **LLM observability** — every query traced in LangFuse with latency, sources, and keyword-recall scores
- **Evaluation discipline** — hand-curated 15-question eval set; 82% avg keyword recall at baseline
- **Full-stack delivery** — FastAPI backend + React frontend + Docker compose + HF Spaces single-container
- **Cost-aware engineering** — Claude Haiku in dev, Sonnet for demos; free-tier Pinecone serverless; local embeddings (HuggingFace MiniLM)

---

## 🏗️ Architecture

```mermaid
flowchart LR
    U[User] -->|POST /chat/stream| FE[React Frontend\nVite · Tailwind]
    FE -->|SSE stream| U
    FE --> BE[FastAPI Backend]
    BE -->|embed query| EMB[all-MiniLM-L6-v2\nlocal CPU]
    BE -->|top-5 chunks| PC[(Pinecone\nServerless)]
    BE -->|prompt + context| CL[Claude\nHaiku / Sonnet]
    BE -->|trace + scores| LF[LangFuse\nObservability]
    PC -->|indexed from| DOCS[99 Anthropic\ndoc pages]

    classDef ext fill:#fef3c7,stroke:#f59e0b
    class EMB,PC,CL,LF,DOCS ext
```

**Request flow:**

1. User sends a question → frontend POST `{messages}` to `/chat/stream`
2. Backend embeds the question with `all-MiniLM-L6-v2` (local, free)
3. Pinecone returns the **top-5 most semantically similar** chunks from 4,788 indexed doc chunks
4. LangChain assembles a grounded prompt with a strict citation-enforcement system message
5. Claude generates an answer using **only** the retrieved context
6. SSE streams tokens to the frontend as they arrive; sources emit as a first event
7. LangFuse records the full trace — input, output, retrieved sources, latency, keyword-recall score

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop
- API keys: [Anthropic](https://console.anthropic.com/), [Pinecone](https://app.pinecone.io/), [LangFuse](https://cloud.langfuse.com/)

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot-template.git
cd rag-chatbot-template
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Launch with Docker Compose

```bash
docker compose up --build
```

This builds the backend (~569 MB, torch-cpu) and frontend (~93 MB, nginx) images and starts both services.

### 3. Index the documentation *(first time only, ~3–5 min)*

```bash
curl -X POST http://localhost:8000/ingest
```

This scrapes 99 Anthropic documentation pages, splits them into ~4,800 chunks, embeds them locally, and upserts to Pinecone.

### 4. Open the chat UI

- **Chat:** http://localhost:5173
- **API docs (Swagger):** http://localhost:8000/docs

---

## 🗂️ Project Structure

```
rag-chatbot-template/
├── backend/
│   ├── app/
│   │   ├── config.py           # Pydantic settings — fail-fast on missing env vars
│   │   ├── main.py             # FastAPI app + CORS + SPA static serving (HF Spaces)
│   │   ├── routers/
│   │   │   ├── chat.py         # POST /chat (sync) + POST /chat/stream (SSE)
│   │   │   └── ingest.py       # POST /ingest
│   │   └── services/
│   │       ├── embeddings.py   # HuggingFace MiniLM singleton
│   │       ├── vectorstore.py  # Pinecone serverless wrapper
│   │       ├── ingest.py       # Scrape → split → embed → upsert (99 doc URLs)
│   │       ├── rag_chain.py    # LCEL pipeline: retriever | prompt | Claude | parser
│   │       └── tracing.py      # LangFuse CallbackHandler factory
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Chat UI — messages, sources, streaming cursor
│   │   ├── useChat.ts          # Hook: SSE consumer + conversation state
│   │   └── types.ts            # Shared TypeScript interfaces
│   └── Dockerfile              # Multi-stage: Node build → nginx serve
├── evaluation/
│   ├── datasets/
│   │   └── eval_questions.json # 15 hand-curated questions with expected keywords
│   └── run_eval.py             # Hits /chat, scores recall, pushes to LangFuse
├── docs/
│   └── architecture.md
├── docker-compose.yml          # Local dev: backend + frontend
├── Dockerfile.hf               # HF Spaces: single-container (FE + BE)
└── PRD.md                      # Product requirements document
```

---

## 🔧 Customization

| To change... | Edit... |
|---|---|
| Source documentation | `backend/app/services/ingest.py` → `DOCS_URLS` list |
| LLM model | `.env` → `CLAUDE_MODEL=claude-sonnet-4-6` |
| Embedding model | `.env` → `EMBEDDING_MODEL` |
| Top-k retrieval | `.env` → `TOP_K` |
| System prompt | `backend/app/services/rag_chain.py` → `SYSTEM_PROMPT` |
| Eval questions | `evaluation/datasets/eval_questions.json` |

---

## 📊 Evaluation

Run the included evaluation harness against a live backend:

```bash
# backend must be running
python evaluation/run_eval.py
```

**V1 baseline results (Claude Haiku 4.5, top-k=5):**

| Metric | Value |
|---|---|
| Questions | 15 (5 lookup · 4 conceptual · 3 multi-doc · 2 refusal · 1 API) |
| Avg keyword recall | **82%** |
| Median keyword recall | 100% |
| Avg latency | ~9.8 s |
| P50 latency | ~9.7 s |

Scores are pushed to LangFuse automatically — visible in the dashboard grouped by trace.

**Known V1 failure modes (planned for V2):**

- *Retrieval gaps:* 2 questions miss because the relevant chunk isn't in top-5. V2 adds Cohere reranker + BM25 hybrid search.
- *Refusal phrasing:* Model refuses off-topic questions correctly but uses synonyms that keyword matching misses. V2 adds LLM-as-judge.

---

## 🛣️ Roadmap

### V2 — RAG Quality Depth
- [ ] Cohere reranker (top-20 → rerank → top-5)
- [ ] Hybrid search (`EnsembleRetriever` BM25 + vector)
- [ ] LLM-as-judge scorer alongside keyword recall
- [ ] Token-level eval metrics (TTFT, tokens/sec)
- [ ] Animated demo GIF + Hugging Face Space live demo

### V3 — Optional Stretch
- [ ] GitHub Actions CI (ruff + eslint + pytest on PR)
- [ ] Multi-doc toggle (Anthropic / FastAPI / dbt indexes)
- [ ] Redis-backed session persistence

---

## 🚢 Deploy to Hugging Face Spaces

```bash
# Build the single-container image locally to test
docker build -f Dockerfile.hf -t rag-chatbot-hf .

# Then push to HF via the Spaces Docker SDK
# See: https://huggingface.co/docs/hub/spaces-sdks-docker
```

The `Dockerfile.hf` builds the React frontend (no `VITE_BACKEND_URL` → relative URLs) and bundles it into the Python image. FastAPI detects `dist/` at startup and serves the SPA alongside the API on port 7860.

---

## 📄 License

MIT © Pablo

---

## 🙌 Stack

[FastAPI](https://fastapi.tiangolo.com) · [LangChain](https://python.langchain.com) · [Pinecone](https://www.pinecone.io) · [Anthropic Claude](https://www.anthropic.com/claude) · [LangFuse](https://langfuse.com) · [React](https://react.dev) · [Vite](https://vitejs.dev) · [Tailwind CSS](https://tailwindcss.com)
