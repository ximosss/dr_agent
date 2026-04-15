"""
Step 4: Convert extracted trajectories to LLaMA-Factory ShareGPT format.

Handles all 4 phases:
  - intent/planning/answer: simple single-turn conversations
  - search: multi-turn with tool calls in Qwen3 hermes format

The output format matches what LLaMA-Factory expects with the `sharegpt` formatting
and `qwen3` template, including proper tool call tags.

Usage:
    python training/convert_sharegpt.py [--input-dir PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAX_TOOL_RESPONSE_CHARS = 4000


# ---------------------------------------------------------------------------
# Tool call format conversion: OpenAI API format → Qwen3 inline format
# ---------------------------------------------------------------------------

def convert_openai_to_qwen3_tool_call(tool_calls: list[dict]) -> str:
    """Convert OpenAI-style tool_calls to inline <tool_call> tags."""
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", tc)  # handle both nested and flat
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        parts.append(
            f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}\n</tool_call>'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Search trajectory conversion
# ---------------------------------------------------------------------------

def convert_search_trajectory(item: dict) -> dict | None:
    """Convert a search trajectory to ShareGPT format with tool calls."""
    msgs = item.get("messages", [])
    tools = item.get("tools", [])

    if len(msgs) < 2:
        return None

    conversations = []

    for msg in msgs:
        role = msg.get("role", "")
        content = str(msg.get("content", "") or "")
        tool_calls = msg.get("tool_calls") or []

        if role == "system":
            conversations.append({"from": "system", "value": content})

        elif role == "user":
            conversations.append({"from": "human", "value": content})

        elif role == "assistant":
            value = content
            if tool_calls:
                tc_text = convert_openai_to_qwen3_tool_call(tool_calls)
                if value and not value.endswith("\n"):
                    value += "\n"
                value += tc_text
            if value.strip():
                conversations.append({"from": "gpt", "value": value})

        elif role == "tool":
            # Truncate long tool responses
            truncated = content[:MAX_TOOL_RESPONSE_CHARS]
            if len(content) > MAX_TOOL_RESPONSE_CHARS:
                truncated += "\n[... truncated]"
            conversations.append({"from": "observation", "value": truncated})

    # Validate: must have system + human + at least one gpt
    roles = [c["from"] for c in conversations]
    if "system" not in roles or "human" not in roles or "gpt" not in roles:
        return None

    # Must end with gpt (final answer, not a pending tool call)
    if conversations[-1]["from"] != "gpt":
        return None

    result = {"conversations": conversations}
    if tools:
        result["tools"] = json.dumps(tools, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# Single-turn conversion (intent / planning / answer)
# ---------------------------------------------------------------------------

def convert_single_turn(item: dict) -> dict | None:
    """Convert a single-turn example to ShareGPT format."""
    system = item.get("system", "")
    user = item.get("user", "")
    assistant = item.get("assistant", "")

    if not user or not assistant:
        return None

    conversations = []
    if system:
        conversations.append({"from": "system", "value": system})
    conversations.append({"from": "human", "value": user})
    conversations.append({"from": "gpt", "value": assistant})

    return {"conversations": conversations}


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def run_conversion(input_dir: Path, output_dir: Path) -> dict:
    """Convert all extracted data to ShareGPT format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    # --- Search trajectories ---
    search_path = input_dir / "search_trajectories.jsonl"
    if search_path.exists():
        search_items = _load_jsonl(search_path)
        converted = []
        for item in search_items:
            result = convert_search_trajectory(item)
            if result:
                converted.append(result)
        _save_json(converted, output_dir / "search.json")
        stats["search"] = len(converted)
        print(f"Converted {len(converted)}/{len(search_items)} search trajectories")
    else:
        stats["search"] = 0
        print(f"No search trajectories found at {search_path}")

    # --- Single-turn phases ---
    for phase in ["intent", "planning", "answer"]:
        phase_path = input_dir / f"{phase}_examples.jsonl"
        if not phase_path.exists():
            stats[phase] = 0
            print(f"No {phase} examples found at {phase_path}")
            continue

        items = _load_jsonl(phase_path)
        converted = []
        for item in items:
            result = convert_single_turn(item)
            if result:
                converted.append(result)
        _save_json(converted, output_dir / f"{phase}.json")
        stats[phase] = len(converted)
        print(f"Converted {len(converted)}/{len(items)} {phase} examples")

    # Save stats
    stats_path = output_dir / "conversion_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nConversion stats: {stats}")
    return stats


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _save_json(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert to ShareGPT format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("training/data/weave_extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/data/sft_ready"),
    )
    args = parser.parse_args()
    run_conversion(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
