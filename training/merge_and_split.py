"""
Step 5: Merge all data sources and create train/val/test splits.

Combines all converted SFT-ready JSON files and creates an explicit train/eval
split for LLaMA-Factory.

Usage:
    python training/merge_and_split.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def save_json(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def run_merge(sft_ready_dir: Path, seed: int = 42, eval_ratio: float = 0.05) -> dict:
    """Merge all converted data and create train/eval splits."""
    all_data = []
    search_data = []
    stats = {"sources": {}}

    # Collect all JSON files in sft_ready
    for json_path in sorted(sft_ready_dir.glob("*.json")):
        if json_path.name in ("train.json", "eval.json", "val.json", "test.json", "merge_stats.json", "conversion_stats.json"):
            continue
        items = load_json(json_path)
        source_name = json_path.stem
        stats["sources"][source_name] = len(items)

        if "search" in source_name:
            search_data.extend(items)
        else:
            all_data.extend(items)

        print(f"  {source_name}: {len(items)} examples")

    # Keep search trajectories at their natural frequency. Earlier versions
    # duplicated search examples when the corpus was small, but that is no
    # longer appropriate once we import thousands of public trajectories.
    search_upsampled = search_data
    stats["search_before_upsample"] = len(search_data)
    stats["search_after_upsample"] = len(search_upsampled)
    all_data.extend(search_upsampled)
    print(f"\nSearch kept at natural count: {len(search_data)}")

    # Shuffle
    random.seed(seed)
    random.shuffle(all_data)

    total = len(all_data)
    n_eval = max(1, int(total * eval_ratio)) if total > 1 else 0
    if n_eval >= total:
        n_eval = max(0, total - 1)
    n_train = total - n_eval

    train_data = all_data[:n_train]
    eval_data = all_data[n_train:]

    save_json(train_data, sft_ready_dir / "train.json")
    save_json(eval_data, sft_ready_dir / "eval.json")

    stats["total"] = total
    stats["train"] = len(train_data)
    stats["eval"] = len(eval_data)

    stats_path = sft_ready_dir / "merge_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nFinal split: train={len(train_data)}, eval={len(eval_data)}")
    print(f"Stats saved to {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge and split SFT data")
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=Path("training/data/sft_ready"),
    )
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    args = parser.parse_args()
    run_merge(args.sft_dir, eval_ratio=args.eval_ratio)


if __name__ == "__main__":
    main()
