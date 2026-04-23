"""
Step 2: Collect teacher model trajectories using the existing agent pipeline.

This script:
1. Loads benchmark questions (FRAMES, SimpleQA, GAIA)
2. Filters out questions already present in Weave data
3. Runs each question through the full agent pipeline using the teacher model
4. Results are automatically captured by Weave (already initialized in agent.py)
5. A dedicated judge agent scores semantic correctness for SFT filtering

Prerequisites:
  - Teacher model must be served via vLLM on the configured endpoint
  - Set env vars: BASE_URL, MODEL_NAME_AT_ENDPOINT to point at teacher

Usage:
    # First update .env to point at teacher model, then:
    python training/collect_teacher.py --benchmark frames --num 40
    python training/collect_teacher.py --benchmark simpleqa --num 20
    python training/collect_teacher.py --benchmark gaia --num 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import load_benchmark, extract_final_answer, score_prediction, EvalExample
from agent import run_autonomous_research, judge_prediction
from training.benchmark_splits import filter_examples_by_partition
from tools import clear_fetch_cache


def load_existing_questions(weave_extracted_dir: Path) -> set[str]:
    """Load question keys already extracted from Weave to avoid duplicates."""
    existing = set()
    search_path = weave_extracted_dir / "search_trajectories.jsonl"
    if search_path.exists():
        with search_path.open() as f:
            for line in f:
                item = json.loads(line)
                existing.add(item.get("question_key", "")[:100])
    return existing


def select_questions(
    benchmark: str,
    num: int,
    existing: set[str],
    text_only: bool = False,
) -> list[EvalExample]:
    """Select questions from the reserved benchmark train split."""
    examples = filter_examples_by_partition(
        benchmark,
        load_benchmark(benchmark),
        partition="train",
    )

    # Filter out existing
    filtered = []
    for ex in examples:
        q_key = ex.question.strip()[:100]
        if q_key in existing:
            continue
        if text_only and ex.file_path:
            continue
        filtered.append(ex)

    # Shuffle and take num
    random.seed(42)
    random.shuffle(filtered)
    selected = filtered[:num]
    print(
        f"[{benchmark}] Train split: {len(examples)}, available after exclusions: {len(filtered)}, "
        f"selected: {len(selected)}"
    )
    return selected


async def collect_trajectories(
    benchmark: str,
    examples: list[EvalExample],
    output_dir: Path,
) -> dict:
    """Run each question through the agent pipeline and record results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"teacher_{benchmark}_{timestamp}.jsonl"

    total = len(examples)
    correct = 0
    errors = 0

    with results_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            print(f"\n[{idx + 1}/{total}] {example.question[:80]}...")
            clear_fetch_cache()

            try:
                prediction_raw, plan = await run_autonomous_research(
                    example.question,
                    local_files_path=example.file_path,
                    eval_mode=True,
                    benchmark_name=benchmark,
                )
                prediction = extract_final_answer(prediction_raw)
                rule_correct = score_prediction(prediction, example.answer)
                judge = await judge_prediction(
                    question=example.question,
                    gold=example.answer,
                    prediction=prediction,
                    prediction_raw=prediction_raw,
                )
                is_correct = judge["correct"]

                if is_correct:
                    correct += 1

                record = {
                    "benchmark": benchmark,
                    "example_id": example.example_id,
                    "question": example.question,
                    "prediction": prediction,
                    "prediction_raw": prediction_raw,
                    "gold": example.answer,
                    "correct": is_correct,
                    "rule_correct": rule_correct,
                    "judge_method": judge.get("method"),
                    "judge_reason": judge.get("reason"),
                    "judge_raw_output": judge.get("raw_output"),
                    "n_objectives": len(plan.objectives) if plan else 0,
                    "collected_sources": [
                        {"objective_id": s["objective_id"], "summary": s["summary"][:500]}
                        for s in (plan.collected_sources if plan else [])
                    ],
                }
                if judge.get("error"):
                    record["judge_error"] = judge["error"]

            except Exception as exc:
                errors += 1
                print(f"  ERROR: {exc}")
                record = {
                    "benchmark": benchmark,
                    "example_id": example.example_id,
                    "question": example.question,
                    "prediction": "",
                    "prediction_raw": "",
                    "gold": example.answer,
                    "correct": False,
                    "error": str(exc),
                }

            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()

    scored = total - errors
    accuracy = correct / scored if scored > 0 else 0
    summary = {
        "benchmark": benchmark,
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
    }
    print(f"\n[{benchmark}] Done: {correct}/{scored} correct ({accuracy:.1%}), {errors} errors")
    return summary


async def main():
    parser = argparse.ArgumentParser(description="Collect teacher trajectories")
    parser.add_argument("-b", "--benchmark", required=True, choices=["frames", "simpleqa", "gaia"])
    parser.add_argument("-n", "--num", type=int, default=40)
    parser.add_argument("--text-only", action="store_true", help="GAIA: skip file-based questions")
    parser.add_argument("--weave-dir", type=Path, default=Path("training/data/weave_extracted"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/teacher_collected"))
    args = parser.parse_args()

    existing = load_existing_questions(args.weave_dir)
    print(f"Found {len(existing)} existing questions to skip")

    text_only = args.text_only or args.benchmark == "gaia"
    questions = select_questions(args.benchmark, args.num, existing, text_only=text_only)

    if not questions:
        print("No new questions to collect!")
        return

    summary = await collect_trajectories(args.benchmark, questions, args.output_dir)
    print(f"\nFinal summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
