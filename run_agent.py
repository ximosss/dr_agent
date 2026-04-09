import argparse
import asyncio
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent import (
    clarify_intent,
    create_search_plan,
    execute_search_plan,
    generate_final_report,
    get_local_context,
    run_autonomous_research,
    run_eval,
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def run_logger(run_type: str) -> str:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{run_type}_{timestamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original_stdout, handle)
        sys.stderr = _Tee(original_stderr, handle)
        try:
            yield str(log_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search", type=str, help="Run a non-interactive search with the given query.")
    group.add_argument("--eval", action="store_true", help="Run evaluation on a benchmark.")
    parser.add_argument("-b", "--benchmark", type=str, choices=["simpleqa", "gaia", "frames"], help="Benchmark name.")
    parser.add_argument("-n", "--num", type=int, default=None, help="Number of examples to evaluate.")
    return parser.parse_args()


async def human_in_the_loop() -> tuple[str, Optional[str]]:
    print("\n" + "=" * 60)
    print("Deep Research Agent")
    print("=" * 60)

    print("\nWhat would you like to research?")
    print("(Enter your question, or 'quit' to exit)")

    research_question = ""
    while not research_question.strip():
        research_question = input("\n> ").strip()
        if research_question.lower() == "quit":
            return "", None

    print("\nDo you have local files to provide as context?")
    print("(Enter path or press Enter to skip)")
    local_files_path = input("> ").strip()

    if local_files_path:
        expanded_path = os.path.expanduser(local_files_path)
        if not os.path.exists(expanded_path):
            print(f"Warning: Path '{local_files_path}' not found. Proceeding without local context.")
            local_files_path = None

    return research_question, local_files_path or None


async def main() -> None:
    args = parse_args()

    if args.search:
        with run_logger("search") as log_path:
            print("\n[Search] Running non-interactive search.")
            final_report, _ = await run_autonomous_research(args.search, eval_mode=False)
            print(f"\n[Search] Log saved to {log_path}")
        return

    if args.eval:
        if not args.benchmark:
            print("Error: --eval requires -b/--benchmark.")
            return
        with run_logger(f"eval_{args.benchmark}") as log_path:
            print(f"\n[Eval] Starting benchmark: {args.benchmark}")
            await run_eval(args.benchmark, args.num)
            print(f"[Eval] Log saved to {log_path}")
        return

    question, local_files_path = await human_in_the_loop()
    if not question:
        print("\nExiting...")
        return

    print("\n[Phase 1] Analyzing local context...")
    local_context = None
    if local_files_path:
        local_context = await get_local_context(local_files_path, question)
        if local_context:
            print(f"  Local context loaded ({len(local_context)} chars)")

    print("\n[Phase 1] Clarifying research intent...")
    clarified_intent = await clarify_intent(question, local_context)

    print("\n" + "-" * 40)
    print("Clarified Intent:")
    print(clarified_intent["clarified_intent"][:500])
    print("-" * 40)

    confirm = input("\nDoes this capture your intent? (y/n/edit): ").strip().lower()
    if confirm == "n":
        print("Exiting. Please refine your question and try again.")
        return
    if confirm == "edit":
        new_intent = input("Enter clarification: ").strip()
        clarified_intent["clarified_intent"] += f"\n\nUser clarification: {new_intent}"

    print("\n[Phase 2] Creating search plan...")
    plan = await create_search_plan(clarified_intent)

    print("\n" + "-" * 40)
    print("Research Plan:")
    print(plan.to_context_string())
    print("-" * 40)

    confirm = input("\nProceed with this plan? (y/n): ").strip().lower()
    if confirm != "y":
        print("Exiting. Please try again with a different approach.")
        return

    print("\n[Phase 3] Executing search plan...")
    plan = await execute_search_plan(plan)

    print("\n[Phase 4] Generating final report...")
    # final_report = await generate_final_report(plan)

    print("\n[End] Thank you for using Deep Research Agent!")


if __name__ == "__main__":
    asyncio.run(main())
