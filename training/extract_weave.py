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
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

TOOL_FAILURE_PATTERNS = [
    (re.compile(r"\[objective_error\]", re.IGNORECASE), "objective_error"),
    (re.compile(r"tool\s+\w+\s+not found in agent", re.IGNORECASE), "tool_not_found"),
    (re.compile(r"\btraceback\b", re.IGNORECASE), "traceback"),
    (re.compile(r"\bexception\b", re.IGNORECASE), "exception"),
    (re.compile(r"\b(?:timed?\s*out|timeout)\b", re.IGNORECASE), "timeout"),
    (re.compile(r"\brate limit(?:ed)?\b", re.IGNORECASE), "rate_limited"),
    (re.compile(r"\b429\b", re.IGNORECASE), "http_429"),
    (re.compile(r"\b(?:500|502|503|504)\b", re.IGNORECASE), "http_5xx"),
    (re.compile(r"\b(?:request|fetch|download|crawl|lookup)\s+(?:failed|failure)\b", re.IGNORECASE), "tool_request_failed"),
    (re.compile(r"\b(?:unable|failed)\s+to\s+(?:fetch|retrieve|open|access)\b", re.IGNORECASE), "fetch_failed"),
    (re.compile(r"\bno (?:content|results?) found\b", re.IGNORECASE), "no_content"),
]

FETCH_WEBPAGE_TOOL_NAMES = {"fetch_webpage", "fetch_web_page"}


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

def load_gold_answers(eval_dir: Path, extra_dirs: list[Path] | None = None) -> dict[str, dict]:
    """Load scored answers keyed by question text (first 100 chars)."""
    gold = {}
    source_dirs = [eval_dir]
    if extra_dirs:
        source_dirs.extend(extra_dirs)

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for jsonl_path in source_dir.glob("*.jsonl"):
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


def get_benchmark_gold_info(
    question_key: str,
    benchmark_train_question_keys: set[str],
    gold: dict[str, dict],
) -> dict:
    """Return benchmark gold info only for reserved-train questions."""
    short_key = question_key[:100]
    if short_key not in benchmark_train_question_keys:
        return {}
    return gold.get(short_key, {})


def is_correct_benchmark_question(
    question_key: str,
    benchmark_train_question_keys: set[str],
    gold: dict[str, dict],
) -> bool:
    """True only when the question is in benchmark-train and scored correct."""
    gold_info = get_benchmark_gold_info(question_key, benchmark_train_question_keys, gold)
    return gold_info.get("correct") is True


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


def stringify_message_content(content) -> str:
    """Best-effort conversion of tool content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def get_tool_name(msg: dict) -> str:
    """Extract a normalized tool name from a tool message if available."""
    for key in ("name", "tool_name"):
        value = msg.get(key)
        if value:
            return str(value).strip().lower()
    return ""


def classify_tool_failure(msg: dict) -> str | None:
    """Return a failure reason when a tool message looks unusable for training."""
    if msg.get("role") != "tool":
        return None

    tool_name = get_tool_name(msg)
    content = stringify_message_content(msg.get("content"))
    stripped = content.strip()

    if not stripped:
        return f"{tool_name or 'tool'}_empty_response"
    if stripped in {"{}", "[]", "null", "None"}:
        return f"{tool_name or 'tool'}_empty_payload"

    for pattern, reason in TOOL_FAILURE_PATTERNS:
        if pattern.search(stripped):
            return f"{tool_name + '_' if tool_name else ''}{reason}"

    if tool_name in FETCH_WEBPAGE_TOOL_NAMES:
        lowered = stripped.lower()
        if lowered.startswith("error"):
            return f"{tool_name}_error_prefix"
        if "failed to fetch" in lowered or "unable to fetch" in lowered:
            return f"{tool_name}_fetch_failed"

    return None


def classify_trajectory_tool_failure(msgs: list[dict]) -> str | None:
    """Return the first detected tool failure reason in a trajectory."""
    for msg in msgs:
        reason = classify_tool_failure(msg)
        if reason:
            return reason
    return None


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
    teacher_dir: Path,
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
    gold = load_gold_answers(eval_dir, extra_dirs=[teacher_dir])
    print(f"Loaded {len(gold)} scored answers from {[str(eval_dir), str(teacher_dir)]}")
    benchmark_train_question_keys = load_partition_question_keys("train")

    stats = {}
    search_filter_stats: dict[str, int] = defaultdict(int)

    # --- Extract search trajectories ---
    search_records = list(by_phase.get("interactive_search", []))
    for record in by_phase.get("eval_search", []):
        msgs = record.get("inputs", {}).get("messages", [])
        if len(msgs) < 2:
            continue
        question_key = extract_question_key(str(msgs[1].get("content", "")))
        if is_correct_benchmark_question(question_key, benchmark_train_question_keys, gold):
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
            search_filter_stats["too_few_tool_responses"] += 1
            continue
        if not traj["has_final_answer"]:
            search_filter_stats["missing_final_answer"] += 1
            continue
        if check_degenerate_loop(traj["messages"]):
            search_filter_stats["degenerate_tool_loop"] += 1
            continue
        tool_failure_reason = classify_trajectory_tool_failure(traj["messages"])
        if tool_failure_reason:
            search_filter_stats[tool_failure_reason] += 1
            continue

        gold_info = get_benchmark_gold_info(q_key, benchmark_train_question_keys, gold)
        if gold_info and gold_info.get("correct") is not True:
            search_filter_stats["benchmark_incorrect"] += 1
            continue

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
    stats["search_filter_stats"] = dict(sorted(search_filter_stats.items()))
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
            gold_info = get_benchmark_gold_info(question_key, benchmark_train_question_keys, gold)
            if not gold_info:
                continue
            if gold_info.get("correct") is not True:
                continue

            # Deduplicate by user message
            user_key = ex["user"][:200]
            if user_key in seen_users:
                continue
            seen_users.add(user_key)

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
    total_examples = sum(value for value in stats.values() if isinstance(value, int))
    print(f"Total: {total_examples} examples")

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
    parser.add_argument(
        "--teacher-dir",
        type=Path,
        default=Path("training/data/teacher_collected"),
        help="Directory containing teacher benchmark results with correct labels",
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

    run_extraction(args.weave_path, args.eval_dir, args.teacher_dir, args.output_dir)


if __name__ == "__main__":
    main()
