"""
Step 1: Extract structured trajectories from Weave JSONL export.

Parses the raw LLM call records into per-phase examples:
  - intent:   single-turn intent clarification
  - planning: single-turn search plan generation
  - search:   multi-turn tool-calling trajectories (reconstructed)
  - answer:   single-turn final answer generation

Usage:
    python training/extract_weave.py [--weave-path PATH] [--eval-dir PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from training.benchmark_splits import load_partition_question_keys


# ---------------------------------------------------------------------------
# Phase classification based on system prompt content
# ---------------------------------------------------------------------------

PHASE_PATTERNS = {
    "eval_search":    "autonomous evaluation research agent",
    "eval_intent":    "preparing an autonomous evaluation run",
    "eval_planning":  "planning assistant for evaluation",
    "eval_answer":    "producing the final answer for an evaluation",
    "interactive_search":  "Deep Research Agent specialized",
    "interactive_intent":  "helping clarify the user's research intent",
    "interactive_planning": "research planning assistant",
    "interactive_polish":   "finalizing a research report",
}


def classify_record(record: dict) -> str:
    msgs = record.get("inputs", {}).get("messages", [])
    if not msgs:
        return "unknown"
    sys_content = str(msgs[0].get("content", ""))
    for phase, pattern in PHASE_PATTERNS.items():
        if pattern in sys_content:
            return phase
    return "unknown"


# ---------------------------------------------------------------------------
# Gold answer matching from eval_outputs/
# ---------------------------------------------------------------------------

def load_gold_answers(eval_dir: Path) -> dict[str, dict]:
    """Load gold answers keyed by question text (first 100 chars)."""
    gold = {}
    for jsonl_path in eval_dir.glob("*.jsonl"):
        with jsonl_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = str(rec.get("question", ""))[:100].strip()
                if q and q not in gold:
                    gold[q] = {
                        "gold_answer": rec.get("gold"),
                        "prediction": rec.get("prediction"),
                        "correct": rec.get("correct"),
                        "benchmark": rec.get("benchmark"),
                        "example_id": rec.get("example_id"),
                    }
    return gold


# ---------------------------------------------------------------------------
# Search trajectory reconstruction
# ---------------------------------------------------------------------------

def extract_question_key(user_msg: str) -> str:
    """Extract a stable key from the user message of a search record."""
    for pattern in (
        r"Research Question:\s*(.+?)(?:\n|$)",
        r"Question:\s*(.+?)(?:\n|$)",
        r"User question:\s*(.+?)(?:\n|$)",
    ):
        match = re.search(pattern, user_msg)
        if match:
            return match.group(1).strip()[:120]
    return user_msg.strip()[:120]


def reconstruct_search_trajectory(record: dict) -> dict:
    """Convert a Weave record (the longest for its question) into a trajectory."""
    msgs = record.get("inputs", {}).get("messages", [])
    output = record.get("output", {})
    tools = record.get("inputs", {}).get("tools", [])

    # Build the full message sequence including the final output
    full_msgs = list(msgs)
    if isinstance(output, dict):
        for choice in output.get("choices", []):
            out_msg = choice.get("message", {})
            if out_msg:
                full_msgs.append(out_msg)

    # Count tool calls
    n_tool_responses = sum(1 for m in full_msgs if m.get("role") == "tool")
    n_think = sum(
        1 for m in full_msgs
        if m.get("role") == "assistant" and "<think>" in str(m.get("content", "") or "")
    )

    # Check if final message is a non-tool-call assistant message
    has_final_answer = True
    if isinstance(output, dict):
        for choice in output.get("choices", []):
            if choice.get("message", {}).get("tool_calls"):
                has_final_answer = False

    return {
        "messages": full_msgs,
        "tools": tools,
        "n_tool_responses": n_tool_responses,
        "n_think_blocks": n_think,
        "has_final_answer": has_final_answer,
    }


def check_degenerate_loop(msgs: list[dict], max_repeats: int = 3) -> bool:
    """Return True if the trajectory has degenerate repeated tool calls."""
    calls = []
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                sig = f"{fn.get('name', '')}|{fn.get('arguments', '')}"
                calls.append(sig)
    if not calls:
        return False
    from collections import Counter
    counts = Counter(calls)
    return any(c >= max_repeats for c in counts.values())


# ---------------------------------------------------------------------------
# Single-turn extraction (intent / planning / answer)
# ---------------------------------------------------------------------------

def extract_single_turn(record: dict) -> dict | None:
    """Extract a single-turn example from a Weave record."""
    msgs = record.get("inputs", {}).get("messages", [])
    output = record.get("output", {})

    if len(msgs) < 2:
        return None

    system_content = str(msgs[0].get("content", ""))
    user_content = str(msgs[1].get("content", ""))

    # Get assistant output
    assistant_content = ""
    if isinstance(output, dict):
        for choice in output.get("choices", []):
            assistant_content = str(choice.get("message", {}).get("content", "") or "")

    if not assistant_content:
        return None

    return {
        "system": system_content,
        "user": user_content,
        "assistant": assistant_content,
    }


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def run_extraction(
    weave_path: Path,
    eval_dir: Path,
    output_dir: Path,
) -> dict:
    """Extract all data from Weave JSONL and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = []
    with weave_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {weave_path}")

    # Classify
    by_phase: dict[str, list] = defaultdict(list)
    for r in records:
        phase = classify_record(r)
        by_phase[phase].append(r)

    print("Records by phase:")
    for phase, recs in sorted(by_phase.items()):
        print(f"  {phase}: {len(recs)}")

    # Load gold answers
    gold = load_gold_answers(eval_dir)
    print(f"Loaded {len(gold)} gold answers from {eval_dir}")
    benchmark_train_question_keys = load_partition_question_keys("train")

    stats = {}

    # --- Extract search trajectories ---
    search_records = list(by_phase.get("interactive_search", []))
    for record in by_phase.get("eval_search", []):
        msgs = record.get("inputs", {}).get("messages", [])
        if len(msgs) < 2:
            continue
        question_key = extract_question_key(str(msgs[1].get("content", "")))
        if question_key[:100] in benchmark_train_question_keys:
            search_records.append(record)
    question_groups: dict[str, list] = defaultdict(list)
    for r in search_records:
        msgs = r.get("inputs", {}).get("messages", [])
        if len(msgs) >= 2:
            user_msg = str(msgs[1].get("content", ""))
            key = extract_question_key(user_msg)
            question_groups[key].append(r)

    search_trajs = []
    for q_key, recs in question_groups.items():
        longest = max(recs, key=lambda r: len(r.get("inputs", {}).get("messages", [])))
        traj = reconstruct_search_trajectory(longest)

        # Apply filters
        if traj["n_tool_responses"] < 2:
            continue
        if not traj["has_final_answer"]:
            continue
        if check_degenerate_loop(traj["messages"]):
            continue

        # Match gold answer
        gold_info = gold.get(q_key[:100], {})

        search_trajs.append({
            "id": f"weave_search_{len(search_trajs):03d}",
            "phase": "search",
            "question_key": q_key,
            "messages": traj["messages"],
            "tools": traj["tools"],
            "n_tool_responses": traj["n_tool_responses"],
            "n_think_blocks": traj["n_think_blocks"],
            "gold_answer": gold_info.get("gold_answer"),
            "is_correct": gold_info.get("correct"),
            "benchmark": gold_info.get("benchmark"),
        })

    _save_jsonl(search_trajs, output_dir / "search_trajectories.jsonl")
    stats["search"] = len(search_trajs)
    print(f"Extracted {len(search_trajs)} search trajectories")

    # --- Extract single-turn phases ---
    for phase_name, phase_key in [
        ("intent", "eval_intent"),
        ("planning", "eval_planning"),
        ("answer", "eval_answer"),
    ]:
        phase_records = by_phase.get(phase_key, [])
        examples = []
        seen_users = set()
        for r in phase_records:
            ex = extract_single_turn(r)
            if ex is None:
                continue

            question_key = extract_question_key(ex["user"])
            if question_key[:100] not in benchmark_train_question_keys:
                continue

            # Deduplicate by user message
            user_key = ex["user"][:200]
            if user_key in seen_users:
                continue
            seen_users.add(user_key)

            # For answer phase, match gold
            gold_info = {}
            if phase_name == "answer":
                q_match = re.search(r"Question:\s*(.+?)(?:\n|$)", ex["user"])
                if q_match:
                    q_text = q_match.group(1).strip()[:100]
                    gold_info = gold.get(q_text, {})

            examples.append({
                "id": f"weave_{phase_name}_{len(examples):03d}",
                "phase": phase_name,
                "system": ex["system"],
                "user": ex["user"],
                "assistant": ex["assistant"],
                "gold_answer": gold_info.get("gold_answer"),
                "is_correct": gold_info.get("correct"),
            })

        _save_jsonl(examples, output_dir / f"{phase_name}_examples.jsonl")
        stats[phase_name] = len(examples)
        print(f"Extracted {len(examples)} {phase_name} examples")

    # Save summary
    summary_path = output_dir / "extraction_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"Total: {sum(stats.values())} examples")

    return stats


def _save_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract trajectories from Weave JSONL")
    parser.add_argument(
        "--weave-path",
        type=Path,
        default=None,
        help="Path to Weave JSONL export (auto-detected if not specified)",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("eval_outputs"),
        help="Directory containing eval output JSONL files with gold answers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/data/weave_extracted"),
        help="Output directory for extracted data",
    )
    args = parser.parse_args()

    # Auto-detect weave path
    if args.weave_path is None:
        candidates = sorted(Path(".").glob("weave_export_*.jsonl"), reverse=True)
        if not candidates:
            print("ERROR: No weave_export_*.jsonl found. Use --weave-path.")
            return
        args.weave_path = candidates[0]
        print(f"Auto-detected Weave export: {args.weave_path}")

    run_extraction(args.weave_path, args.eval_dir, args.output_dir)


if __name__ == "__main__":
    main()
