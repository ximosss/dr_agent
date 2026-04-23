"""
Download traced LLM calls from Weave and save them as local JSONL.

The output format is designed to match what training/extract_weave.py expects:
each line is a Weave Call.to_dict() record containing inputs / output / summary.

Usage:
    uv run python training/download_weave_export.py
    uv run python training/download_weave_export.py --output-path weave_export_latest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import weave


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT
DEFAULT_PROJECT = "ximo_ml/deep_research_agent"
DEFAULT_FETCH_LIMIT = 50000
DEFAULT_OP_SUBSTRINGS = ("litellm.acompletion", "litellm.completion")
CALL_COLUMNS = [
    "id",
    "trace_id",
    "parent_id",
    "op_name",
    "inputs",
    "output",
    "exception",
    "summary",
    "attributes",
    "started_at",
    "ended_at",
    "display_name",
    "thread_id",
    "turn_id",
    "wb_run_id",
    "wb_run_step",
]


def status_name(record: dict[str, Any]) -> str:
    status = ((record.get("summary") or {}).get("weave") or {}).get("status")
    if status is None:
        return ""
    return str(status).split(".")[-1].lower()


def looks_like_llm_message_call(record: dict[str, Any], op_substrings: tuple[str, ...]) -> bool:
    op_name = str(record.get("op_name") or "")
    if not any(substr in op_name for substr in op_substrings):
        return False
    inputs = record.get("inputs") or {}
    return isinstance(inputs.get("messages"), list)


def is_complete_call(record: dict[str, Any]) -> bool:
    if record.get("output") is not None:
        return True
    return status_name(record) in {"success", "completed"}


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"weave_export_{timestamp}.jsonl"


def download_weave_calls(
    project: str,
    output_path: Path,
    fetch_limit: int,
    include_incomplete: bool,
    op_substrings: tuple[str, ...],
) -> dict[str, int]:
    client = weave.init(project, settings={"print_call_link": False})
    calls = client.get_calls(
        limit=fetch_limit,
        sort_by=[{"field": "started_at", "direction": "desc"}],
        columns=CALL_COLUMNS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fetched = 0
    matched = 0
    written = 0
    statuses: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for call in calls:
            fetched += 1
            record = call.to_dict()
            status = status_name(record)
            statuses[status] = statuses.get(status, 0) + 1

            if not looks_like_llm_message_call(record, op_substrings):
                continue
            matched += 1

            if not include_incomplete and not is_complete_call(record):
                continue

            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            written += 1

    return {
        "fetched": fetched,
        "matched_llm_calls": matched,
        "written": written,
        **{f"status_{key or 'unknown'}": value for key, value in sorted(statuses.items())},
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Download Weave traces to a local JSONL export")
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT") or DEFAULT_PROJECT,
        help="Weave/W&B project name, e.g. entity/project",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path(),
        help="Where to write the downloaded JSONL export",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=DEFAULT_FETCH_LIMIT,
        help="Maximum number of calls to fetch from Weave",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include calls that have not completed successfully yet",
    )
    parser.add_argument(
        "--op-substring",
        action="append",
        default=list(DEFAULT_OP_SUBSTRINGS),
        help="Only keep calls whose op_name contains this substring (repeatable)",
    )
    args = parser.parse_args()

    stats = download_weave_calls(
        project=args.project,
        output_path=args.output_path,
        fetch_limit=args.fetch_limit,
        include_incomplete=args.include_incomplete,
        op_substrings=tuple(args.op_substring),
    )

    print(f"Project: {args.project}")
    print(f"Output: {args.output_path}")
    print(json.dumps(stats, indent=2))

    if stats["written"] == 0:
        raise SystemExit("No completed Weave calls were exported.")


if __name__ == "__main__":
    main()
