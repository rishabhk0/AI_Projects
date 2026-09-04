"""
Eval harness for the Grounded LLM Router.

Runs eval_questions.json (a mix of answerable and deliberately unanswerable
questions) through the full pipeline and reports:
  - Grounded-answer rate on answerable questions (did it answer when it should?)
  - Correct-refusal rate on unanswerable questions (did it refuse when it should?)
  - Avg cost and latency per query

Failed generations (API errors that survived all retries) are excluded from
scoring rather than counted as wrong — they measure API reliability, not the
router's grounding logic. See results/eval_results.csv for the last run this
was actually scored on.

Usage:
    python eval.py
"""

import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from core import GroundedRouter

load_dotenv()

# Spacing between calls — free-tier RPM limits are the real bottleneck here,
# not anything in this code. Lower this at your own risk.
SECONDS_BETWEEN_CALLS = 5


def run_eval(api_key: str | None = None, out_csv: str = "results/eval_results.csv"):
    questions = json.loads(Path("eval_questions.json").read_text())
    router = GroundedRouter(api_key=api_key)

    rows = []
    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}] {q['question'][:60]}")
        result = router.answer(q["question"])
        rows.append(
            {
                "question": q["question"],
                "expected_answerable": q["answerable"],
                "actual_grounded": result["grounded"],
                "generation_failed": result["generation_failed"],
                "correct": (
                    q["answerable"] == result["grounded"]
                    if not result["generation_failed"] else None
                ),
                "answer": result["answer"][:200],
                "model_used": result["model_used"],
                "best_distance": round(result["best_distance"], 3),
                "est_cost_usd": round(result["est_cost_usd"], 6),
                "latency_ms": round(result["latency_ms"], 0),
            }
        )
        time.sleep(SECONDS_BETWEEN_CALLS)

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)

    scored = df[~df.generation_failed]
    failed_count = int(df.generation_failed.sum())
    answerable = scored[scored.expected_answerable]
    unanswerable = scored[~scored.expected_answerable]

    grounded_rate = answerable.actual_grounded.mean() * 100 if len(answerable) else 0
    refusal_rate = (~unanswerable.actual_grounded).mean() * 100 if len(unanswerable) else 0
    overall_accuracy = scored.correct.mean() * 100 if len(scored) else 0
    avg_cost = scored.est_cost_usd.mean() if len(scored) else 0
    avg_latency = scored.latency_ms.mean() if len(scored) else 0

    print("\n=== Eval Results ===")
    print(f"Failed generations (excluded): {failed_count}")
    print(f"Answerable questions:      {len(answerable)}")
    print(f"Unanswerable questions:    {len(unanswerable)}")
    print(f"Grounded-answer rate:      {grounded_rate:.1f}%")
    print(f"Correct-refusal rate:      {refusal_rate:.1f}%")
    print(f"Overall accuracy:          {overall_accuracy:.1f}%")
    print(f"Avg est. cost/query:       ${avg_cost:.6f}")
    print(f"Avg latency:               {avg_latency:.0f} ms")
    print(f"\nFull results written to {out_csv}")

    return df


if __name__ == "__main__":
    run_eval()
