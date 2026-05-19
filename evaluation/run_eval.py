"""End-to-end evaluation harness.

Runs every question in `datasets/eval_questions.json` against the running
backend's `/chat` endpoint, computes keyword recall + latency for each
answer, prints a per-question table, writes a timestamped JSON result
file, and pushes per-question scores to LangFuse so they show up next
to the corresponding traces in the dashboard.

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

    Why this metric: cheap, deterministic, and easy to explain. It misses
    semantic correctness — a model that copies keywords without
    understanding still scores high. PRD V2 adds LLM-as-judge on top to
    catch that class of failures.
    """
    if not expected:
        return 0.0
    a = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in a)
    return hits / len(expected)


def call_chat(question: str) -> dict:
    """POST /chat with a single-turn message and return the JSON response."""
    resp = requests.post(
        f"{BACKEND_URL}/chat",
        json={"messages": [{"role": "user", "content": question}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    print(f"\nRunning {len(questions)} eval questions against {BACKEND_URL}")
    print(f"{'#':>3}  {'category':<10}  {'recall':>6}  {'lat ms':>6}  question")
    print("-" * 90)

    results = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            chat_resp = call_chat(q["question"])
        except Exception as e:
            # Print and keep going — one failed call shouldn't kill the run.
            print(f"{i:>3}  {q['category']:<10}  ERROR    {type(e).__name__}: {str(e)[:60]}")
            continue
        latency_ms = int((time.perf_counter() - t0) * 1000)
        recall = keyword_recall(chat_resp["answer"], q["expected_keywords"])

        # Push both numbers to LangFuse so they appear next to the trace.
        # Scores are how LangFuse plots quality over time in the dashboard.
        lf.score(trace_id=chat_resp["trace_id"], name="keyword_recall", value=recall)
        lf.score(trace_id=chat_resp["trace_id"], name="latency_ms", value=latency_ms)

        results.append({
            "category": q["category"],
            "question": q["question"],
            "expected_keywords": q["expected_keywords"],
            "answer": chat_resp["answer"],
            "recall": recall,
            "latency_ms": latency_ms,
            "trace_id": chat_resp["trace_id"],
        })
        print(f"{i:>3}  {q['category']:<10}  {recall:>6.2f}  {latency_ms:>6}  {q['question'][:60]}")

    # Make sure all scores actually leave the local buffer before we exit.
    lf.flush()

    if not results:
        print("\nNo results — backend probably not running.")
        return 1

    # --- aggregate metrics -----------------------------------------------
    recalls = [r["recall"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    summary = {
        "n_questions": len(results),
        "avg_recall": round(statistics.mean(recalls), 3),
        "median_recall": round(statistics.median(recalls), 3),
        "avg_latency_ms": int(statistics.mean(latencies)),
        "p50_latency_ms": int(statistics.median(latencies)),
        # P95 needs at least 20 samples to be meaningful — fall back to max.
        "p95_latency_ms": (
            int(statistics.quantiles(latencies, n=20)[18])
            if len(latencies) >= 20 else max(latencies)
        ),
    }

    print("\n=== Summary ===")
    print(f"  Questions       : {summary['n_questions']}")
    print(f"  Avg recall      : {summary['avg_recall']:.2%}")
    print(f"  Median recall   : {summary['median_recall']:.2%}")
    print(f"  Avg latency     : {summary['avg_latency_ms']} ms")
    print(f"  P50 latency     : {summary['p50_latency_ms']} ms")
    print(f"  P95 latency     : {summary['p95_latency_ms']} ms")

    # Per-category breakdown — useful to spot if e.g. refusals are failing.
    cats: dict[str, list[float]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["recall"])
    print("\n=== Per category ===")
    for cat, recs in sorted(cats.items()):
        print(f"  {cat:<10}  n={len(recs):<2}  avg_recall={statistics.mean(recs):.2%}")

    # --- save full results to disk ---------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_file = RESULTS_DIR / f"eval-{ts}.json"
    out_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nResults saved to: {out_file.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
