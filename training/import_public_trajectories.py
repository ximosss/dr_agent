"""
Import public Deep Research trajectories into the local search-trajectory format.

This script rewrites external trajectory datasets into the same intermediate
JSONL schema produced by `training.extract_weave`, so the imported examples can
flow through the existing augmentation, ShareGPT conversion, and merge steps.

Currently supported sources:
  - deepresearch-traj
  - edr-200

Usage:
    python -m training.import_public_trajectories \
        --output-path training/data/weave_extracted/search_trajectories.jsonl \
        --deepresearch-dir /data//deepresearch-traj \
        --edr-dir /data/edr-200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from prompt import (
    EVAL_SYSTEM_PROMPT,
    FETCH_WEBPAGE_TOOL_PROMPT,
    LOCAL_DOCS_LOOKUP_TOOL_PROMPT,
    PAPER_SEARCH_TOOL_PROMPT,
    WEB_SEARCH_TOOL_PROMPT,
)


DEFAULT_MAX_FINDINGS_CHARS = 3000


def _build_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": WEB_SEARCH_TOOL_PROMPT,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_webpage",
                "description": FETCH_WEBPAGE_TOOL_PROMPT,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 8000},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "paper_search",
                "description": PAPER_SEARCH_TOOL_PROMPT,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                        "mode": {"type": "string", "default": "precise"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "local_docs_lookup",
                "description": LOCAL_DOCS_LOOKUP_TOOL_PROMPT,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "local_files_path": {"type": "string"},
                        "question": {"type": "string"},
                        "max_chars": {"type": "integer", "default": 12000},
                    },
                    "required": ["local_files_path", "question"],
                },
            },
        },
    ]


TOOLS_SCHEMA = _build_tools_schema()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            return str(text)
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _strip_note_lines(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("note:"):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def _infer_research_type(question: str) -> str:
    lowered = question.lower()
    long_form_markers = (
        "report",
        "survey",
        "analyze",
        "analysis",
        "compare",
        "collect",
        "summarize",
        "综述",
        "分析",
        "研究",
        "报告",
        "整理",
        "比较",
    )
    if len(question) > 180 or any(marker in lowered for marker in long_form_markers):
        return "long-form"
    return "short-form"


def _infer_mode(research_type: str) -> str:
    return "broad" if research_type == "long-form" else "precise"


def _infer_keywords(question: str) -> list[str]:
    if re.search(r"[\u4e00-\u9fff]", question):
        compact = re.sub(r"\s+", " ", question).strip()
        return [compact[:80]] if compact else []

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'._-]*", question)
    seen: set[str] = set()
    keywords: list[str] = []
    for word in words:
        normalized = word.strip("._-")
        if len(normalized) < 3:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(normalized)
        if len(keywords) >= 8:
            break
    if keywords:
        return keywords
    compact = re.sub(r"\s+", " ", question).strip()
    return [compact[:80]] if compact else []


def _build_user_prompt(question: str, research_type: str, mode: str, description: str) -> str:
    keywords = _infer_keywords(question)
    keywords_text = ", ".join(keywords) if keywords else question[:80]
    return f"""
Current Research Plan (including results from completed objectives):
Research Question: {question}
Type: {research_type}

Search Plan:
  [~] #1 [high] {description}

Your current task is Objective #1:
- Description: {description}
- Search type: web
- Mode: {mode}
- Suggested keywords: {keywords_text}

Budget for this objective:
- web_search: at most 8 calls
- fetch_webpage: at most 8 calls
- paper_search: at most 2 calls
- local_docs_lookup: at most 1 calls
- total tool calls across this objective: at most 18

Execute this search objective using the appropriate tool.
Report ONLY the raw facts you found (exact names, numbers, dates, quotes from sources).
Do NOT compute a final answer or draw conclusions — a separate agent will do that.
Stop searching as soon as you have enough evidence to answer the current objective.
""".strip()


def _stable_question_key(question: str) -> str:
    normalized = _normalize_ws(question)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    preview = normalized[:96]
    return f"{preview}::{digest}"


def _resolve_open_url(args: dict, tool_text: str) -> str | None:
    raw_id = args.get("id")
    if isinstance(raw_id, str) and raw_id.startswith(("http://", "https://")):
        return raw_id

    for pattern in (
        r"URL:\s*(https?://\S+)",
        r"Doc \d+ \((https?://[^)]+)\)",
        r"`(https?://[^`]+)`",
    ):
        match = re.search(pattern, tool_text)
        if match:
            return match.group(1).strip()
    return None


def _extract_find_url(tool_text: str) -> str | None:
    url = _resolve_open_url({}, tool_text)
    if not url:
        return None
    return url.split("/find?pattern=")[0]


def _extract_json_like_field(text: str, field: str) -> str | None:
    for quote in ('"', "'"):
        key = f"{quote}{field}{quote}"
        key_index = text.find(key)
        if key_index == -1:
            continue

        colon_index = text.find(":", key_index + len(key))
        if colon_index == -1:
            continue

        i = colon_index + 1
        length = len(text)
        while i < length and text[i].isspace():
            i += 1
        if i >= length:
            return None

        value_quote = text[i]
        if value_quote in {'"', "'"}:
            i += 1
            chars: list[str] = []
            escaped = False
            while i < length:
                ch = text[i]
                if escaped:
                    chars.append(ch)
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == value_quote:
                    break
                else:
                    chars.append(ch)
                i += 1
            return "".join(chars).strip()

        end = i
        while end < length and text[end] not in ",}":
            end += 1
        return text[i:end].strip()

    return None


def _coerce_tool_args(raw_text: str):
    text = raw_text.strip()
    if not text:
        return None

    candidate = text
    for _ in range(3):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            nested_text = _extract_text(parsed).strip()
            if not nested_text or nested_text == candidate:
                break
            candidate = nested_text
            continue
        if isinstance(parsed, str):
            nested_text = parsed.strip()
            if not nested_text or nested_text == candidate:
                return parsed
            candidate = nested_text
            continue
        return parsed

    args: dict[str, object] = {}
    query_value = _extract_json_like_field(candidate, "query")
    if query_value is not None:
        query_value = query_value.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t")
        args["query"] = query_value.strip()

    raw_id = _extract_json_like_field(candidate, "id")
    if raw_id is not None:
        try:
            args["id"] = int(raw_id)
        except ValueError:
            args["id"] = raw_id.replace('\\"', '"').replace("\\'", "'").strip()

    for field in ("topn", "cursor", "loc", "max_results"):
        raw_value = _extract_json_like_field(candidate, field)
        if raw_value is None:
            continue
        try:
            args[field] = int(raw_value)
        except ValueError:
            continue

    if args:
        return args

    return candidate


def _make_tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _extract_deepresearch_findings(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    match = re.search(
        r"Explanation:\s*(.*?)(?:\n\s*Exact Answer:|\n\s*Confidence:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        findings = match.group(1).strip()
        if findings:
            return findings

    text = re.sub(r"(?im)^\s*Exact Answer:.*$", "", text)
    text = re.sub(r"(?im)^\s*Confidence:.*$", "", text)
    text = re.sub(r"(?im)^\s*Explanation:\s*", "", text)
    return _normalize_ws(text)


def _running_report_delta(previous: str, current: str, max_chars: int) -> str:
    current = current.strip()
    if not current:
        return ""

    if previous and current.startswith(previous):
        delta = current[len(previous):].strip()
        if len(delta) >= 120:
            return delta[:max_chars]

    return current[:max_chars]


def _format_edr_search_result(query: str, result: dict) -> str:
    sources = result.get("sources") or []
    if not sources:
        return f"[NO_RESULTS] No results found for query: {query}"

    lines = ["[Search results from imported EDR]\n"]
    for idx, source in enumerate(sources, 1):
        title = str(source.get("title") or source.get("name") or "No title").strip()
        url = str(source.get("url") or source.get("link") or "N/A").strip()
        lines.append(f"{idx}. **{title}**\n   URL: {url}\n")
    return "\n".join(lines)


def _build_deepresearch_trajectory(
    row: dict,
    *,
    max_tool_responses: int,
    max_findings_chars: int,
    source_index: int,
) -> dict | None:
    question = _strip_note_lines(str(row.get("question", "")).strip())
    if not question:
        return None

    research_type = _infer_research_type(question)
    user_prompt = _build_user_prompt(
        question=question,
        research_type=research_type,
        mode=_infer_mode(research_type),
        description="Find the raw evidence needed to answer the research question accurately.",
    )

    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    original_messages = row.get("messages") or []
    pending_tool_name: str | None = None
    tool_count = 0
    think_count = 0
    skipped_tool = False
    final_findings = ""

    for idx, original in enumerate(original_messages):
        role = str(original.get("role", ""))
        recipient = original.get("recipient")
        channel = original.get("channel")
        text = _normalize_ws(_extract_text(original.get("content")))

        if role == "assistant" and recipient:
            pending_tool_name = None
            skipped_tool = False
            if tool_count >= max_tool_responses:
                break

            call_id = f"deepresearch_{source_index}_{idx}"
            if recipient == "browser.search":
                args = _coerce_tool_args(text)
                if isinstance(args, str):
                    query = args.strip()
                    if not query:
                        skipped_tool = True
                        continue
                    max_results = 10
                elif isinstance(args, dict):
                    query = args.get("query")
                    if not isinstance(query, str) or not query.strip():
                        skipped_tool = True
                        continue
                    max_results = int(args.get("topn", args.get("max_results", 10)) or 10)
                else:
                    skipped_tool = True
                    continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call(
                                "web_search",
                                {"query": query.strip(), "max_results": max_results},
                                call_id,
                            )
                        ],
                    }
                )
                pending_tool_name = "web_search"
                continue

            if recipient == "browser.open":
                args = _coerce_tool_args(text)
                if not isinstance(args, dict):
                    skipped_tool = True
                    continue
                next_text = ""
                if idx + 1 < len(original_messages):
                    next_text = _normalize_ws(_extract_text(original_messages[idx + 1].get("content")))
                url = _resolve_open_url(args, next_text)
                if not url:
                    skipped_tool = True
                    continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call(
                                "fetch_webpage",
                                {"url": url, "max_chars": 8000},
                                call_id,
                            )
                        ],
                    }
                )
                pending_tool_name = "fetch_webpage"
                continue

            if recipient == "browser.find":
                next_text = ""
                if idx + 1 < len(original_messages):
                    next_text = _normalize_ws(_extract_text(original_messages[idx + 1].get("content")))
                url = _extract_find_url(next_text)
                if not url:
                    skipped_tool = True
                    continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            _make_tool_call(
                                "fetch_webpage",
                                {"url": url, "max_chars": 8000},
                                call_id,
                            )
                        ],
                    }
                )
                pending_tool_name = "fetch_webpage"
                continue

            skipped_tool = True
            continue

        if role == "tool":
            if not pending_tool_name or skipped_tool:
                continue
            if tool_count >= max_tool_responses:
                break
            messages.append(
                {
                    "role": "tool",
                    "name": pending_tool_name,
                    "content": text,
                }
            )
            tool_count += 1
            pending_tool_name = None
            skipped_tool = False
            continue

        if role != "assistant" or recipient:
            continue

        if channel == "final":
            cleaned = _extract_deepresearch_findings(text)
            if cleaned:
                final_findings = cleaned[:max_findings_chars]
            continue

        if not text:
            continue
        if "<think>" in text:
            think_count += 1
        messages.append({"role": "assistant", "content": text[:max_findings_chars]})

    if final_findings:
        messages.append({"role": "assistant", "content": final_findings})
    elif messages and messages[-1]["role"] == "tool":
        messages.append({"role": "assistant", "content": "Collected the available source evidence for this objective."})

    if tool_count < 2:
        return None
    if messages[-1]["role"] != "assistant":
        return None

    return {
        "id": f"public_deepresearch_{source_index:05d}",
        "phase": "search",
        "question_key": _stable_question_key(question),
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "n_tool_responses": tool_count,
        "n_think_blocks": think_count,
        "gold_answer": None,
        "is_correct": True,
        "benchmark": "public_deepresearch_traj",
    }


def _build_edr_iteration_intro(step: dict, max_chars: int) -> str:
    subtopics: list[str] = []
    gaps: list[str] = []

    for call in step.get("tool_calls", []):
        fn = call.get("function", {})
        if fn.get("name") != "decompose_query":
            continue
        arguments = fn.get("arguments") or {}
        knowledge_gap = str(arguments.get("knowledge_gap", "")).strip()
        if knowledge_gap:
            gaps.append(knowledge_gap)
        for subtopic in (call.get("result") or {}).get("subtopics", []):
            if isinstance(subtopic, dict):
                name = str(subtopic.get("name") or subtopic.get("query") or "").strip()
            else:
                name = str(subtopic or "").strip()
            if name:
                subtopics.append(name)

    parts = []
    if gaps:
        parts.append(f"Need to close this knowledge gap: {gaps[0]}")
    if subtopics:
        parts.append("Search focus: " + "; ".join(subtopics[:4]))

    text = "\n".join(parts).strip()
    if not text:
        return ""
    return text[:max_chars]


def _build_edr_trajectory(
    row: dict,
    *,
    max_tool_responses: int,
    max_searches_per_iteration: int,
    max_findings_chars: int,
    source_index: int,
) -> dict | None:
    question = _strip_note_lines(str(row.get("query", "")).strip())
    if not question:
        return None

    research_type = _infer_research_type(question)
    user_prompt = _build_user_prompt(
        question=question,
        research_type=research_type,
        mode=_infer_mode(research_type),
        description="Collect authoritative web evidence and structured findings needed for the report.",
    )

    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        trajectory = json.loads(str(row.get("trajectory", "")))
    except json.JSONDecodeError:
        return None

    tool_count = 0
    previous_report = ""

    for iteration, step in enumerate(trajectory):
        if tool_count >= max_tool_responses:
            break

        intro = _build_edr_iteration_intro(step, max_findings_chars)
        if intro:
            messages.append({"role": "assistant", "content": intro})

        search_calls = []
        search_results = []
        for call in step.get("tool_calls", []):
            fn = call.get("function", {})
            if fn.get("name") != "general_search":
                continue
            search_calls.append(call)
            if len(search_calls) >= max_searches_per_iteration:
                break

        if search_calls:
            tool_calls = []
            for idx, call in enumerate(search_calls):
                arguments = call.get("function", {}).get("arguments") or {}
                query = str(arguments.get("query", "")).strip()
                if not query:
                    continue
                tool_calls.append(
                    _make_tool_call(
                        "web_search",
                        {"query": query, "max_results": 5},
                        f"edr_{source_index}_{iteration}_{idx}",
                    )
                )
                search_results.append((query, call.get("result") or {}))

            if tool_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                for query, result in search_results:
                    if tool_count >= max_tool_responses:
                        break
                    messages.append(
                        {
                            "role": "tool",
                            "name": "web_search",
                            "content": _format_edr_search_result(query, result),
                        }
                    )
                    tool_count += 1

        current_report = str(step.get("running_report", "") or "")
        delta = _running_report_delta(previous_report, current_report, max_findings_chars)
        if delta:
            messages.append({"role": "assistant", "content": delta})
        previous_report = current_report

    if tool_count < 2:
        return None
    if messages[-1]["role"] != "assistant":
        fallback = _running_report_delta("", previous_report, max_findings_chars)
        if not fallback:
            return None
        messages.append({"role": "assistant", "content": fallback})

    return {
        "id": f"public_edr_{source_index:04d}",
        "phase": "search",
        "question_key": _stable_question_key(question),
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "n_tool_responses": tool_count,
        "n_think_blocks": 0,
        "gold_answer": None,
        "is_correct": None,
        "benchmark": "public_edr_200",
    }


def _load_deepresearch_rows(dataset_dir: Path, *, shuffle_buffer_size: int = 0, shuffle_seed: int = 42):
    parquet_files = sorted(str(path) for path in dataset_dir.glob("*.parquet"))
    if not parquet_files:
        return []
    dataset = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return dataset


def _load_edr_rows(dataset_dir: Path):
    parquet_files = sorted(str(path) for path in dataset_dir.glob("*.parquet"))
    if not parquet_files:
        return []
    return load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)


def run_import(args) -> dict:
    output_path = args.output_path
    existing = _load_jsonl(output_path)
    existing_question_keys = {str(item.get("question_key", "")).strip() for item in existing}
    seen_public_question_keys: set[str] = set()
    imported: list[dict] = []

    stats = {
        "existing_search_trajectories": len(existing),
        "deepresearch": {
            "accepted": 0,
            "skipped_incorrect": 0,
            "skipped_pass_rate": 0,
            "skipped_duplicate": 0,
            "skipped_convert": 0,
        },
        "edr": {"accepted": 0, "skipped_duplicate": 0, "skipped_convert": 0},
    }

    if args.deepresearch_dir and Path(args.deepresearch_dir).exists():
        dataset = _load_deepresearch_rows(
            Path(args.deepresearch_dir),
            shuffle_buffer_size=args.deepresearch_shuffle_buffer,
            shuffle_seed=args.deepresearch_shuffle_seed,
        )
        for row_idx, row in enumerate(dataset):
            if row_idx > 0 and row_idx % 1000 == 0:
                print(
                    f"[deepresearch] scanned={row_idx} accepted={stats['deepresearch']['accepted']} "
                    f"skipped_incorrect={stats['deepresearch']['skipped_incorrect']} "
                    f"skipped_pass_rate={stats['deepresearch']['skipped_pass_rate']} "
                    f"skipped_duplicate={stats['deepresearch']['skipped_duplicate']} "
                    f"skipped_convert={stats['deepresearch']['skipped_convert']}",
                    flush=True,
                )
            if args.max_deepresearch > 0 and stats["deepresearch"]["accepted"] >= args.max_deepresearch:
                break
            if str(row.get("status", "")) != "success" or not bool(row.get("correct")):
                stats["deepresearch"]["skipped_incorrect"] += 1
                continue
            if float(row.get("pass_rate", 0.0) or 0.0) < args.min_deepresearch_pass_rate:
                stats["deepresearch"]["skipped_pass_rate"] += 1
                continue
            item = _build_deepresearch_trajectory(
                row,
                max_tool_responses=args.max_tool_responses,
                max_findings_chars=args.max_findings_chars,
                source_index=row_idx,
            )
            if not item:
                stats["deepresearch"]["skipped_convert"] += 1
                continue
            if item["question_key"] in existing_question_keys or (
                args.dedupe_public_questions and item["question_key"] in seen_public_question_keys
            ):
                stats["deepresearch"]["skipped_duplicate"] += 1
                continue
            imported.append(item)
            if args.dedupe_public_questions:
                seen_public_question_keys.add(item["question_key"])
            stats["deepresearch"]["accepted"] += 1
            if stats["deepresearch"]["accepted"] % 500 == 0:
                print(
                    f"[deepresearch] accepted={stats['deepresearch']['accepted']} "
                    f"scanned={row_idx + 1} skipped_incorrect={stats['deepresearch']['skipped_incorrect']} "
                    f"skipped_pass_rate={stats['deepresearch']['skipped_pass_rate']} "
                    f"skipped_duplicate={stats['deepresearch']['skipped_duplicate']} "
                    f"skipped_convert={stats['deepresearch']['skipped_convert']}",
                    flush=True,
                )

    if args.edr_dir and Path(args.edr_dir).exists():
        dataset = _load_edr_rows(Path(args.edr_dir))
        for row_idx, row in enumerate(dataset):
            if row_idx > 0 and row_idx % 50 == 0:
                print(
                    f"[edr] scanned={row_idx} accepted={stats['edr']['accepted']} "
                    f"skipped_duplicate={stats['edr']['skipped_duplicate']} "
                    f"skipped_convert={stats['edr']['skipped_convert']}",
                    flush=True,
                )
            if args.max_edr > 0 and stats["edr"]["accepted"] >= args.max_edr:
                break
            item = _build_edr_trajectory(
                row,
                max_tool_responses=args.max_tool_responses,
                max_searches_per_iteration=args.max_edr_searches_per_iteration,
                max_findings_chars=args.max_findings_chars,
                source_index=row_idx,
            )
            if not item:
                stats["edr"]["skipped_convert"] += 1
                continue
            if item["question_key"] in existing_question_keys or (
                args.dedupe_public_questions and item["question_key"] in seen_public_question_keys
            ):
                stats["edr"]["skipped_duplicate"] += 1
                continue
            imported.append(item)
            if args.dedupe_public_questions:
                seen_public_question_keys.add(item["question_key"])
            stats["edr"]["accepted"] += 1
            if stats["edr"]["accepted"] % 50 == 0:
                print(
                    f"[edr] accepted={stats['edr']['accepted']} scanned={row_idx + 1} "
                    f"skipped_duplicate={stats['edr']['skipped_duplicate']} "
                    f"skipped_convert={stats['edr']['skipped_convert']}",
                    flush=True,
                )

    merged = existing + imported
    _save_jsonl(merged, output_path)

    summary = {
        **stats,
        "imported_total": len(imported),
        "final_search_trajectories": len(merged),
        "output_path": str(output_path),
    }

    summary_path = output_path.parent / "public_import_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Import public trajectories into search_trajectories.jsonl")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("training/data/weave_extracted/search_trajectories.jsonl"),
    )
    parser.add_argument("--deepresearch-dir", type=Path, default=None)
    parser.add_argument("--edr-dir", type=Path, default=None)
    parser.add_argument("--min-deepresearch-pass-rate", type=float, default=0.5)
    parser.add_argument("--max-deepresearch", type=int, default=0, help="0 means no limit")
    parser.add_argument("--deepresearch-shuffle-buffer", type=int, default=5000)
    parser.add_argument("--deepresearch-shuffle-seed", type=int, default=42)
    parser.add_argument("--dedupe-public-questions", action="store_true")
    parser.add_argument("--max-edr", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-tool-responses", type=int, default=20)
    parser.add_argument("--max-edr-searches-per-iteration", type=int, default=4)
    parser.add_argument("--max-findings-chars", type=int, default=DEFAULT_MAX_FINDINGS_CHARS)
    args = parser.parse_args()
    run_import(args)


if __name__ == "__main__":
    main()
