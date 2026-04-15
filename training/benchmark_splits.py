from __future__ import annotations

import json
import random
from pathlib import Path

from dotenv import load_dotenv

from evals import EvalExample, load_benchmark


BENCHMARKS = ("frames", "simpleqa", "gaia")
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SPLITS_PATH = Path("training/data/benchmark_splits.json")
QUESTION_KEY_LEN = 100

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def make_question_key(question: str) -> str:
    return str(question).strip()[:QUESTION_KEY_LEN]


def _serialize_examples(examples: list[EvalExample]) -> list[dict[str, str]]:
    rows = []
    for example in examples:
        rows.append(
            {
                "example_id": str(example.example_id),
                "question_key": make_question_key(example.question),
            }
        )
    rows.sort(key=lambda item: (item["example_id"], item["question_key"]))
    return rows


def build_benchmark_splits(
    seed: int = DEFAULT_SPLIT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> dict:
    data: dict[str, object] = {
        "seed": seed,
        "test_ratio": test_ratio,
        "benchmarks": {},
    }

    for benchmark in BENCHMARKS:
        rows = _serialize_examples(load_benchmark(benchmark))
        rng = random.Random(f"{seed}:{benchmark}")
        rng.shuffle(rows)

        if len(rows) <= 1:
            train_rows = rows
            test_rows = []
        else:
            n_test = max(1, int(len(rows) * test_ratio))
            if n_test >= len(rows):
                n_test = len(rows) - 1
            test_rows = rows[:n_test]
            train_rows = rows[n_test:]

        data["benchmarks"][benchmark] = {
            "counts": {
                "all": len(rows),
                "train": len(train_rows),
                "test": len(test_rows),
            },
            "train": train_rows,
            "test": test_rows,
        }

    return data


def ensure_benchmark_splits(
    path: Path = DEFAULT_SPLITS_PATH,
    seed: int = DEFAULT_SPLIT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)

    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_benchmark_splits(seed=seed, test_ratio=test_ratio)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def filter_examples_by_partition(
    benchmark: str,
    examples: list[EvalExample],
    partition: str,
    path: Path = DEFAULT_SPLITS_PATH,
) -> list[EvalExample]:
    if partition == "all":
        return examples

    splits = ensure_benchmark_splits(path=path)
    benchmark_data = splits["benchmarks"][benchmark]
    allowed_ids = {row["example_id"] for row in benchmark_data[partition]}
    allowed_questions = {row["question_key"] for row in benchmark_data[partition]}

    filtered = []
    for example in examples:
        example_id = str(example.example_id)
        question_key = make_question_key(example.question)
        if example_id and example_id in allowed_ids:
            filtered.append(example)
        elif not example_id and question_key in allowed_questions:
            filtered.append(example)

    return filtered


def load_partition_question_keys(
    partition: str,
    path: Path = DEFAULT_SPLITS_PATH,
) -> set[str]:
    splits = ensure_benchmark_splits(path=path)
    keys: set[str] = set()
    for benchmark in BENCHMARKS:
        keys.update(row["question_key"] for row in splits["benchmarks"][benchmark][partition])
    return keys
