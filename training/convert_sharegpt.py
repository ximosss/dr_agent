"""
Step 4: Convert extracted trajectories to LLaMA-Factory ShareGPT format.

Handles all 4 phases:
  - intent/planning/answer: simple single-turn conversations
  - search: multi-turn with tool calls in Qwen3 hermes format

The output format matches what LLaMA-Factory expects with the `sharegpt` formatting
and `qwen3` template, including proper tool call tags.

Usage:
    python training/convert_sharegpt.py [--weave-dir PATH] [--augmented-dir PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
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
    pending_assistant_text = ""
    idx = 0
    while idx < len(msgs):
        msg = msgs[idx]
        role = msg.get("role", "")
        content = str(msg.get("content", "") or "")
        tool_calls = msg.get("tool_calls") or []

        if role == "system":
            conversations.append({"from": "system", "value": content})
            idx += 1
            continue

        if role == "user":
            conversations.append({"from": "human", "value": content})
            idx += 1
            continue

        if role == "assistant":
            if tool_calls:
                tc_text = convert_openai_to_qwen3_tool_call(tool_calls)
                function_parts = [part for part in [pending_assistant_text.strip(), content.strip(), tc_text] if part]
                if function_parts:
                    conversations.append({"from": "function", "value": "\n\n".join(function_parts)})
                pending_assistant_text = ""

                idx += 1
                observations = []
                while idx < len(msgs) and msgs[idx].get("role", "") == "tool":
                    tool_content = str(msgs[idx].get("content", "") or "")
                    truncated = tool_content[:MAX_TOOL_RESPONSE_CHARS]
                    if len(tool_content) > MAX_TOOL_RESPONSE_CHARS:
                        truncated += "\n[... truncated]"
                    if truncated.strip():
                        observations.append(truncated)
                    idx += 1

                if observations:
                    conversations.append({"from": "observation", "value": "\n\n".join(observations)})
                continue

            if content.strip():
                pending_assistant_text = "\n\n".join(
                    [part for part in [pending_assistant_text.strip(), content.strip()] if part]
                )
            idx += 1
            continue

        if role == "tool":
            truncated = content[:MAX_TOOL_RESPONSE_CHARS]
            if len(content) > MAX_TOOL_RESPONSE_CHARS:
                truncated += "\n[... truncated]"
            if truncated.strip():
                conversations.append({"from": "observation", "value": truncated})
            idx += 1
            continue

        idx += 1

    if pending_assistant_text.strip():
        conversations.append({"from": "gpt", "value": pending_assistant_text})

    # Validate: must have system + human + at least one gpt
    roles = [c["from"] for c in conversations]
    if "system" not in roles or "human" not in roles or "gpt" not in roles:
        return None

    # Must end with gpt (final answer, not a pending tool call)
    if conversations[-1]["from"] != "gpt":
        return None

    result = {"conversations": conversations, "tools": ""}
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

    return {"conversations": conversations, "tools": ""}


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def _convert_search_file(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0

    items = _load_jsonl(path)
    converted = []
    for item in items:
        result = convert_search_trajectory(item)
        if result:
            converted.append(result)
    return converted, len(items)


def _convert_single_turn_file(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        return [], 0

    items = _load_jsonl(path)
    converted = []
    for item in items:
        result = convert_single_turn(item)
        if result:
            converted.append(result)
    return converted, len(items)


def run_conversion(weave_dir: Path, augmented_dir: Path, output_dir: Path) -> dict:
    """Convert extracted + augmented data to ShareGPT format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"sources": {}}

    search_data, search_total = _convert_search_file(weave_dir / "search_trajectories.jsonl")
    _save_json(search_data, output_dir / "search.json")
    stats["search"] = len(search_data)
    stats["sources"]["search"] = {"converted": len(search_data), "raw": search_total}
    print(f"Converted {len(search_data)}/{search_total} search trajectories")

    single_turn_sources = {
        "intent": [
            weave_dir / "intent_examples.jsonl",
            augmented_dir / "intent_examples.jsonl",
        ],
        "planning": [
            weave_dir / "planning_examples.jsonl",
            augmented_dir / "planning_examples.jsonl",
        ],
        "answer": [
            weave_dir / "answer_examples.jsonl",
            augmented_dir / "answer_examples.jsonl",
        ],
        "search_reuse": [
            augmented_dir / "search_reuse_examples.jsonl",
        ],
    }

    for phase, paths in single_turn_sources.items():
        converted = []
        raw_total = 0
        for path in paths:
            phase_converted, phase_raw = _convert_single_turn_file(path)
            converted.extend(phase_converted)
            raw_total += phase_raw
        _save_json(converted, output_dir / f"{phase}.json")
        stats[phase] = len(converted)
        stats["sources"][phase] = {"converted": len(converted), "raw": raw_total}
        print(f"Converted {len(converted)}/{raw_total} {phase} examples")

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
        "--weave-dir",
        type=Path,
        default=Path("training/data/weave_extracted"),
    )
    parser.add_argument(
        "--augmented-dir",
        type=Path,
        default=Path("training/data/augmented"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/data/sft_ready"),
    )
    args = parser.parse_args()
    run_conversion(args.weave_dir, args.augmented_dir, args.output_dir)


if __name__ == "__main__":
    main()
