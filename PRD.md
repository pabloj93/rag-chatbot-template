# PRD — `rag-chatbot-template`

> **Author:** Pablo (pablo@rapidcanvas.ai)
> **Date:** 2026-05-15
> **Status:** Draft v1.0 — to be ratified before any code is written
> **Estimated build:** 15–20 hours

---

## 1. Project Overview — *Logical Thinking*

### What
A production-grade, open-source **RAG chatbot template** that answers questions about the **Anthropic / Claude documentation** using Claude itself. Shipped as a clean, well-documented GitHub repository with a live demo on Hugging Face Spaces.

### Why
This repository serves as a **technical portfolio piece** to attract **AI / LLM Engineer roles** at AI-first companies (Anthropic, OpenAI, scale-ups, ML-heavy startups). It demonstrates end-to-end mastery of the modern RAG stack while making a memorable meta-statement: *"I build with Claude, I understand Claude deep enough to RAG-index its docs, and I produce code that is shippable, observable, and evaluated."*

The repo must withstand a **30-second skim** by a recruiter (visual polish, screenshots, demo link, stack badges) **and** a **20-minute deep-read** by a senior engineer (clean architecture, documented trade-offs, evaluation metrics, observability).

### Success Criteria
1. **Recruiter signal** — README scannable in 30s; live demo opens in ≤10s; ≥3 visual elements (architecture diagram, screenshots, animated GIF).
2. **Engineer signal** — code is idiomatic, modular, with explicit retrieval/eval choices documented; LangFuse traces visible publicly or via screenshot.
3. **Functional bar** — answers on Anthropic docs are grounded (≥80% keyword recall on eval set), cite sources, and latency P95 < 4s with Haiku / < 8s with Sonnet.
4. **Discoverability** — repo ranks for "claude rag template" / "anthropic rag chatbot" GitHub search within 30 days of publish.

### Non-Goals (explicit out-of-scope for this build)
- Multi-tenant / auth / billing
- **Server-side** conversation persistence. History lives in the frontend (browser session) and is sent with each `/chat` request; the backend stays stateless so it survives HF Spaces sleeps and multi-replica deploys.
- Production-grade rate-limiting, SSO, RBAC
- Mobile-native app
- Fine-tuning or model training of any kind

---

## 2. Skills & Tools — *Analytical Thinking*

### Languages
- **Python 3.11+** — backend, ingestion, evaluation
- **TypeScript (React 18)** — frontend
- **Bash / PowerShell** — scripting & deploy commands

### Core Frameworks
| Layer | Tool | Why this one |
|-------|------|--------------|
| Backend API | **FastAPI 0.115** | Async, type-safe, auto OpenAPI, industry standard for AI APIs |
| RAG orchestration | **LangChain 0.3** | Composable retrievers/chains; signal to hiring managers |
| Vector DB | **Pinecone serverless** | Free tier sufficient; reads well on a resume; production-grade |
| LLM | **Claude Sonnet 4.6** (prod) / **Haiku 4.5** (dev) | Meta-fit with the docs being indexed; budget-aware via Haiku |
| Embeddings | **HuggingFace `all-MiniLM-L6-v2`** (local) | Free, fast, good enough for 384-dim retrieval |
| Reranking (V2) | **Cohere Rerank v3** | Free trial tier; well-known improvement on naive cosine |
| Observability | **LangFuse Cloud** | Free tier, traces + scores, recruiter-visible dashboard screenshots |
| Frontend | **React 18 + Vite + TypeScript + Tailwind** (minimal, demo-only) | Just enough React to prove the architecture works end-to-end. No shadcn, no design system, no animations. The frontend is a **thin demo shell**, not a portfolio piece on its own. |
| Charts | **Recharts** (only if a metric panel adds clear signal) | Lighter than Plotly, React-native, no JS↔Python bridge. Add only if it strengthens the RAG-quality story. |
| Containerization | **Docker + docker-compose** | One-command local dev; standard expectation |
| Demo hosting | **Hugging Face Spaces** (Docker SDK) | Free, persistent URL, ML-community visible |

### Infrastructure
- **Hugging Face Space** (Docker SDK) — hosts the full stack (FE + BE in one image, or split via separate Spaces)
- **GitHub** — repo, Actions for lint + test on PR
- **Pinecone serverless** — 1 index, ~5k vectors (well under free-tier 100k limit)
- **LangFuse Cloud** — free tier (50k traces/month, more than enough)
- **Local dev** — Docker Desktop on Windows 11

### API Keys & Costs (budget: $5–10 total for the build)
| Service | Cost during build | Notes |
|---------|-------------------|-------|
| Anthropic | ~$2–5 | Haiku for all dev iterations; Sonnet only for final demo recording + eval runs |
| Pinecone | $0 | Serverless free tier |
| LangFuse | $0 | Free tier |
| HuggingFace | $0 | Spaces free tier |
| Cohere (V2) | $0 | Free trial tier |
| **Total** | **~$5** | |

### Skills the Project Demonstrates
- RAG architecture (chunking, embedding, retrieval, generation)
- Prompt engineering (system prompts, grounding, citation enforcement)
- LLM observability (LangFuse traces, scores, datasets)
- Evaluation discipline (eval datasets, automated keyword recall, latency tracking)
- Full-stack delivery (FastAPI + React + Docker)
- DevEx (one-command setup, clean README, architecture diagrams)
- Cost-aware engineering (Haiku/Sonnet split, free-tier choices)

---

## 3. Key Features — *Computational Thinking*

Roadmap is **strictly incremental**: V1 must ship before V2 begins. V1 carries the differentiator (RAG quality); V2 polishes UX; V3 is bonus for after the repo is already public.

### V1 — Core RAG Engine (10–12h, the MVP)
Goal: prove the system answers Anthropic-docs questions accurately, with citations, observable end-to-end.

**Backend**
- [ ] FastAPI app with `/health`, `/ingest`, `/chat` endpoints
- [ ] Ingestion script that scrapes ~30 Anthropic doc pages (prompt caching, tool use, vision, messages API, model cards, etc.)
- [ ] `RecursiveCharacterTextSplitter` with 800/120 chunk/overlap
- [ ] HuggingFace embeddings → Pinecone serverless
- [ ] LangChain RAG chain: retriever (top-k=5) → prompt → Claude Sonnet 4.6
- [ ] System prompt enforces: ONLY-context answers, mandatory citation block
- [ ] LangFuse tracing on every `/chat` call (input, output, sources, latency)
- [ ] Frontend-led conversation history — `/chat` accepts the last N turns sent by the client (industry-standard pattern à la Anthropic Messages API); backend remains stateless

**Frontend (intentionally minimal — demo shell only)**
- [ ] React + Vite + TS + Tailwind scaffold (no shadcn, no UI library)
- [ ] Single chat page: input box, message list, sources rendered as plain `<details>` blocks
- [ ] SSE streaming via `EventSource` — single `/chat/stream` endpoint pushes tokens as they arrive; non-streaming `/chat` kept as fallback for tools that can't SSE
- [ ] Loading / error / empty states using Tailwind utility classes only
- [ ] No animations, no dark mode, no responsive polish beyond Tailwind defaults
- [ ] Total frontend code budget: **~250 LOC** (streaming state machine adds ~50 LOC over plain fetch). If it grows past that, the architecture is wrong for V1.

**Evaluation**
- [ ] Hand-curated `eval_questions.json` (15 questions on Anthropic docs)
- [ ] `run_eval.py` that hits `/chat`, scores keyword recall, pushes scores to LangFuse
- [ ] Screenshot of LangFuse dashboard saved in `docs/screenshots/`

**Infra & Docs**
- [ ] `docker-compose.yml` brings up FE + BE
- [ ] README with stack badges, architecture Mermaid diagram, quick-start
- [ ] `.env.example` complete; secrets never committed

**V1 Definition of Done:** A reviewer can `git clone`, fill `.env`, run `docker compose up`, hit ingest, ask 5 questions, and see grounded answers with sources in under 5 minutes from clone.

---

### V2 — RAG Quality Depth (4–5h)
Goal: lift RAG quality so the meta-story ("AI/LLM Engineer who measures") is unambiguous. **Frontend stays minimal** — V2 is backend depth, not UI polish.

- [ ] **Cohere reranker** layered on top of Pinecone retrieval (top-20 → rerank → top-5)
- [ ] **Hybrid search** (BM25 + vector) via LangChain `EnsembleRetriever`
- [ ] **Persistent multi-session conversations** — Redis-backed session store (TTL ~7 days), so a user can resume a thread across browser sessions. Keeps the V1 stateless contract intact by treating Redis as an *optional* cache for client-pulled history.
- [ ] **Token-level eval metrics** — measure time-to-first-token (TTFT) and tokens-per-second across the eval dataset (streaming already shipped in V1, so we now measure its quality)
- [ ] **Expanded eval set** (40 questions, grouped by topic) + retrieval-precision metric alongside recall
- [ ] **Animated demo GIF** at top of README (30s screen recording via ScreenToGif)
- [ ] **Hugging Face Space deploy** — live demo URL published in README
- [ ] **CONTRIBUTING.md** + MIT license

---

### V3 — Optional Stretch (only if V1+V2 land within budget) (2–3h)
- [ ] GitHub Actions: lint (ruff + eslint), pytest on backend, build on PR
- [ ] Dark mode toggle (shadcn theme)
- [ ] Multi-doc support (toggle between Anthropic / FastAPI / dbt indexes)
- [ ] OpenTelemetry export from LangFuse for advanced observability blog post

---

## 4. Audience & Constraints — *Procedural Thinking*

### Primary Audience: AI / LLM Engineer Recruiters & Hiring Managers
**Where they encounter the repo:** LinkedIn portfolio link, "Featured projects" on GitHub profile, recruiter outreach response, application attachment.

**Their typical scan path (30 seconds):**
1. Top of README → tagline, stack badges, animated GIF
2. Architecture diagram → "do they understand systems?"
3. "Try it live" link → click → does it work in 10s?
4. Quick scroll to "Features" / "Roadmap" → "is this real or vibes-coded?"
5. *Maybe* glance at `backend/app/services/rag_chain.py` → "is the code clean?"

**Implication:** Every one of those 5 steps must deliver a positive signal. There is no second chance.

### Secondary Audience: Senior Engineers Evaluating Architecture
**Where they encounter it:** Hiring manager forwards link, technical interviewer reviews before screen, peer-review on a PR they're helping with.

**Their scan path (20 minutes):**
1. README → architecture doc → start asking "why this choice?"
2. Read `rag_chain.py`, `vectorstore.py`, `ingest.py`
3. Look at `evaluation/` — "do they measure?"
4. Check `docs/architecture.md` for trade-off discussion
5. Skim git history — "did this person work iteratively?"

**Implication:** Each major design choice (chunk size, top-k, system prompt, eval metric) needs a one-line rationale in code comments or `docs/`. Git commits must be atomic and well-titled.

### Specific Experiences This Audience Has
- They have seen **dozens of "ChatGPT clone" projects** with no eval, no observability, no production thinking. The bar is differentiation.
- They are **skeptical of LangChain bloat** — code must use LangChain idiomatically, not as a crutch hiding a 3-line retrieval call.
- They know **what good RAG looks like** — citations, grounding, eval scores, retrieval-aware system prompts. Half-measures will be spotted.
- They have **5 tabs open** when they look at the repo. The README must hold attention against that competition.

### Constraints

**Technical**
- **Budget cap:** $10 total spend across all APIs during build → forces Haiku-in-dev / Sonnet-in-demo discipline
- **Free tier limits:** Pinecone 100k vectors (we'll use ~5k), HF Spaces 16GB RAM / sleeps after 48h inactivity, LangFuse 50k traces/month
- **Windows-first dev environment:** all scripts must work in PowerShell; Docker Desktop dependency
- **No paid hosting:** demo lives on HF Spaces with cold-start tolerance (~30s wake-up acceptable)

**Time**
- **15–20h hard cap** — V3 is optional, V1 must ship even if V2 slips
- **Calendar:** target public push within 3 weeks of start

**Audience-driven**
- **English-only documentation** — international recruiter audience
- **Accessibility (basic):** alt text on screenshots, semantic HTML in React, keyboard-navigable chat input
- **Mobile-friendly demo:** the chat page must work on a phone (recruiters open links on mobile)
- **No vendor lock-in framing:** README must show that Pinecone / Claude can be swapped — recruiters at competing labs will read this repo

**Stylistic**
- **No emoji-spam in code or commits** — emojis allowed in README headers only, for scannability
- **Commits are atomic and conventional** (`feat:`, `fix:`, `docs:`, `refactor:`)
- **No AI-generated-looking prose in README** — short, declarative sentences; no "In today's fast-paced world…"

### Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| HF Space cold-start makes demo feel broken | High | Add a banner: "First load takes ~30s while the Space wakes up" |
| Anthropic docs scraping breaks (HTML changes) | Medium | Save raw HTML to `data/` as a snapshot; document refresh process |
| LangChain version churn breaks examples | Medium | Pin exact versions; CI runs weekly to catch breakage |
| Budget overrun on Claude | Low | All dev iteration on Haiku; Sonnet only for final eval + demo recording |
| Frontend eats time meant for RAG quality | High | **Hard rule: frontend is a demo shell. No UI library, no shadcn, no animations, no dark mode. ~250 LOC cap. If you catch yourself styling, stop and go back to retrieval.** |
| Recruiter dismisses repo as "ugly UI" | Medium | README leads with **architecture diagram + LangFuse screenshot + eval metrics** — not the UI. The UI is framed in the README as "intentionally minimal demo shell" so it reads as a choice, not a gap. |
| SSE streaming hits CORS/proxy issues on HF Spaces | Medium | Test deploy at end of V1 (not at the very end of the project). Keep non-streaming `/chat` endpoint alive as a fallback. Set `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers on the SSE response. |

---

## 📋 Sign-off Checklist (before writing code)

- [ ] Project Overview reviewed — Pablo agrees with positioning as "AI/LLM Eng portfolio, meta Anthropic-docs RAG"
- [ ] Skills & Tools reviewed — no surprise dependencies, budget understood
- [ ] V1 scope reviewed — Pablo can articulate the V1 DoD in one sentence
- [ ] Audience constraints reviewed — Pablo accepts that the React frontend is a **demo shell only** (~200 LOC, no UI library, no polish) and the differentiation lives in the backend RAG quality
- [ ] PRD committed to repo as `PRD.md` before first line of application code
