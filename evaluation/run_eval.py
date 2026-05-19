"""End-to-end evaluation harness.

Runs every question in `datasets/eval_questions.json` against the running
backend's `/chat/stream` endpoint. Measures three metrics per question:

  - keyword_recall   : fraction of expected keywords in the answer (quality)
  - ttft_ms          : time from request to first SSE token (UX / retrieval speed)
  - chunks_per_sec   : SSE token-chunks per second after the first chunk (gen speed)

NOTE: `chunks_per_sec` counts SSE chunks, not model tokens. Claude batches
multiple tokens per SSE event, so the real token rate is higher. This metric
is still useful for relative comparison across versions.

All three scores are pushed to LangFuse and written to a timestamped JSON.

Usage (backend must be running on http://localhost:8000):

    python evaluation/run_eval.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from langfuse import Langfuse


# .env at the project root holds LangFuse + (optionally) BACKEND_URL.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_PATH = Path(__file__).parent / "datasets" / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def keyword_recall(answer: str, expected: list[str]) -> float:
    """Fraction of expected keywords present (case-insensitive) in `answer`.

    Why this metric: cheap, deterministic, easy to explain. A model that
    copies keywords without understanding still scores high — V3 will add
    LLM-as-judge on top to catch that class of failure.
    """
    if not expected:
        return 0.0
    a = answer.lower()
    return sum(1 for kw in expected if kw.lower() in a) / len(expected)


def call_chat_stream(question: str) -> dict:
    """POST /chat/stream and consume the SSE response.

    Returns a dict with:
        answer          — full answer text (accumulated from token events)
        sources         — list of {url, snippet} from the sources event
        trace_id        — LangFuse trace ID from the done event
        latency_ms      — total client-measured latency
        ttft_ms         — time from request to first token chunk (TTFT)
        chunk_count     — number of SSE token-chunks received
        chunks_per_sec  — chunk_count / (latency - ttft) in seconds

    Why measure TTFT from the client side: it includes retrieval + embedding
    time, which is the dominant cost before Claude starts generating. A high
    TTFT means retrieval is slow; a low TTFT with high total latency means
    the model is slow to generate.
    """
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BACKEND_URL}/chat/stream",
        json={"messages": [{"role": "user", "content": question}]},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    ttft_ms: float | None = None
    answer_parts: list[str] = []
    sources: list[dict] = []
    trace_id = ""
    current_event = ""

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")

        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            payload = json.loads(line[6:])
            if current_event == "token":
                # Record TTFT on the very first token chunk.
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                answer_parts.append(payload["text"])
            elif current_event == "sources":
                sources = payload
            elif current_event == "done":
                trace_id = payload.get("trace_id", "")
            elif current_event == "error":
                raise RuntimeError(payload.get("message", "SSE error"))

    total_ms = int((time.perf_counter() - t0) * 1000)
    ttft_ms_int = int(ttft_ms) if ttft_ms is not None else 0
    chunk_count = len(answer_parts)

    # Chunks per second = chunks generated / generation time (after first chunk).
    generation_s = max((total_ms - ttft_ms_int) / 1000, 0.001)
    chunks_per_sec = round(chunk_count / generation_s, 2) if chunk_count > 1 else 0.0

    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "trace_id": trace_id,
        "latency_ms": total_ms,
        "ttft_ms": ttft_ms_int,
        "chunk_count": chunk_count,
        "chunks_per_sec": chunks_per_sec,
    }


def main() -> int:
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    print(f"\nRunning {len(questions)} eval questions against {BACKEND_URL}")
    print(f"{'#':>3}  {'category':<10}  {'recall':>6}  {'lat ms':>6}  {'ttft':>5}  {'cps':>5}  question")
    print("-" * 100)

    results = []
    for i, q in enumerate(questions, 1):
        try:
            r = call_chat_stream(q["question"])
        except Exception as e:
            print(f"{i:>3}  {q['category']:<10}  ERROR  {type(e).__name__}: {str(e)[:60]}")
            continue

        recall = keyword_recall(r["answer"], q["expected_keywords"])

        # Push all three metrics to LangFuse so they show on the trace timeline.
        lf.score(trace_id=r["trace_id"], name="keyword_recall", value=recall)
        lf.score(trace_id=r["trace_id"], name="ttft_ms", value=r["ttft_ms"])
        lf.score(trace_id=r["trace_id"], name="chunks_per_sec", value=r["chunks_per_sec"])

        results.append({
            "category": q["category"],
            "question": q["question"],
            "expected_keywords": q["expected_keywords"],
            "answer": r["answer"],
            "recall": recall,
            "latency_ms": r["latency_ms"],
            "ttft_ms": r["ttft_ms"],
            "chunk_count": r["chunk_count"],
            "chunks_per_sec": r["chunks_per_sec"],
            "trace_id": r["trace_id"],
        })
        print(
            f"{i:>3}  {q['category']:<10}  {recall:>6.2f}"
            f"  {r['latency_ms']:>6}  {r['ttft_ms']:>5}  {r['chunks_per_sec']:>5.1f}"
            f"  {q['question'][:55]}"
        )

    lf.flush()

    if not results:
        print("\nNo results — backend probably not running.")
        return 1

    # --- aggregate -----------------------------------------------------------
    recalls = [r["recall"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0]
    cpss = [r["chunks_per_sec"] for r in results if r["chunks_per_sec"] > 0]

    summary = {
        "n_questions": len(results),
        "avg_recall": round(statistics.mean(recalls), 3),
        "median_recall": round(statistics.median(recalls), 3),
        "avg_latency_ms": int(statistics.mean(latencies)),
        "p50_latency_ms": int(statistics.median(latencies)),
        "p95_latency_ms": (
            int(statistics.quantiles(latencies, n=20)[18])
            if len(latencies) >= 20 else max(latencies)
        ),
        "avg_ttft_ms": int(statistics.mean(ttfts)) if ttfts else 0,
        "avg_chunks_per_sec": round(statistics.mean(cpss), 2) if cpss else 0.0,
    }

    print("\n=== Summary ===")
    print(f"  Questions          : {summary['n_questions']}")
    print(f"  Avg recall         : {summary['avg_recall']:.2%}")
    print(f"  Median recall      : {summary['median_recall']:.2%}")
    print(f"  Avg latency        : {summary['avg_latency_ms']} ms")
    print(f"  P50 latency        : {summary['p50_latency_ms']} ms")
    print(f"  P95 latency        : {summary['p95_latency_ms']} ms")
    print(f"  Avg TTFT           : {summary['avg_ttft_ms']} ms  (time to first token)")
    print(f"  Avg chunks/sec     : {summary['avg_chunks_per_sec']}  (SSE chunks, not raw tokens)")

    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["recall"])
    print("\n=== Per category ===")
    for cat, recs in sorted(cats.items()):
        print(f"  {cat:<10}  n={len(recs):<2}  avg_recall={statistics.mean(recs):.2%}")

    # --- save ----------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_file = RESULTS_DIR / f"eval-{ts}.json"
    out_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nResults saved to: {out_file.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
