"""
Step 3: Synthetic data augmentation using the teacher model.

Generates additional training examples WITHOUT consuming Tavily API calls:
  1. Synthetic intent clarification (from diverse benchmark questions)
  2. Synthetic planning (from generated intents)
  3. Synthetic answer generation (using real search results from collected trajectories)
  4. Observation-reuse augmentation (alternative reasoning over same tool responses)

Prerequisites:
  - Teacher model served via vLLM at the configured endpoint
  - Existing search trajectories in training/data/weave_extracted/

Usage:
    python training/augment_data.py [--num-intent 150] [--num-answer 100] [--num-reuse 40]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

load_dotenv()

import os

from prompt import (
    EVAL_INTENT_CLARIFICATION_PROMPT,
    EVAL_PLANNING_PROMPT,
    EVAL_ANSWER_PROMPT,
    EVAL_SYSTEM_PROMPT,
)
from evals import load_benchmark


def create_model():
    return LitellmModel(
        model="hosted_vllm/" + os.getenv("MODEL_NAME_AT_ENDPOINT"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("BASE_KEY"),
    )


def strip_think_block(text: str) -> str:
    import re
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text)


# ---------------------------------------------------------------------------
# Collect diverse questions from benchmarks
# ---------------------------------------------------------------------------

def collect_questions(num: int) -> list[str]:
    """Collect diverse questions from all benchmarks."""
    questions = []
    for bench in ["frames", "simpleqa"]:
        try:
            examples = load_benchmark(bench)
            questions.extend([ex.question for ex in examples])
        except Exception as e:
            print(f"Warning: could not load {bench}: {e}")
    try:
        examples = load_benchmark("gaia")
        questions.extend([ex.question for ex in examples if not ex.file_path])
    except Exception as e:
        print(f"Warning: could not load gaia: {e}")

    random.seed(42)
    random.shuffle(questions)
    return questions[:num]


# ---------------------------------------------------------------------------
# 1. Synthetic Intent Generation
# ---------------------------------------------------------------------------

async def generate_intents(questions: list[str], output_path: Path) -> list[dict]:
    """Generate intent clarifications for each question."""
    agent = Agent(
        name="IntentGenerator",
        instructions=EVAL_INTENT_CLARIFICATION_PROMPT,
        model=create_model(),
    )

    results = []
    for i, q in enumerate(questions):
        try:
            result = await Runner.run(agent, f"User question: {q}", max_turns=1)
            output = strip_think_block(result.final_output)
            results.append({
                "id": f"aug_intent_{i:03d}",
                "phase": "intent",
                "system": EVAL_INTENT_CLARIFICATION_PROMPT,
                "user": f"User question: {q}",
                "assistant": result.final_output,  # keep think blocks for training
            })
            if (i + 1) % 20 == 0:
                print(f"  Intent: {i + 1}/{len(questions)}")
        except Exception as e:
            print(f"  Intent {i} failed: {e}")

    _save_jsonl(results, output_path)
    print(f"Generated {len(results)} intent examples")
    return results


# ---------------------------------------------------------------------------
# 2. Synthetic Planning Generation
# ---------------------------------------------------------------------------

async def generate_plans(intents: list[dict], output_path: Path) -> list[dict]:
    """Generate search plans from intent outputs."""
    agent = Agent(
        name="PlanGenerator",
        instructions=EVAL_PLANNING_PROMPT,
        model=create_model(),
    )

    results = []
    for i, intent in enumerate(intents):
        question = intent["user"].replace("User question: ", "")
        clarified = strip_think_block(intent["assistant"])

        message = f"""
Based on this research intent, create a search plan:

Question: {question}

Clarified Intent:
{clarified}

Local Context Available: No

Output a JSON array of search objectives.
"""
        try:
            result = await Runner.run(agent, message, max_turns=1)
            results.append({
                "id": f"aug_planning_{i:03d}",
                "phase": "planning",
                "system": EVAL_PLANNING_PROMPT,
                "user": message,
                "assistant": result.final_output,
            })
            if (i + 1) % 20 == 0:
                print(f"  Planning: {i + 1}/{len(intents)}")
        except Exception as e:
            print(f"  Planning {i} failed: {e}")

    _save_jsonl(results, output_path)
    print(f"Generated {len(results)} planning examples")
    return results


# ---------------------------------------------------------------------------
# 3. Synthetic Answer Generation
# ---------------------------------------------------------------------------

async def generate_answers(
    search_trajs: list[dict],
    output_path: Path,
    num: int = 100,
) -> list[dict]:
    """Generate final answers using real collected sources."""
    agent = Agent(
        name="AnswerGenerator",
        instructions=EVAL_ANSWER_PROMPT,
        model=create_model(),
    )

    # Build (question, sources) pairs from search trajectories
    pairs = []
    for traj in search_trajs:
        q_key = traj.get("question_key", "")
        # Extract sources from assistant messages in the trajectory
        msgs = traj.get("messages", [])
        sources = []
        for m in msgs:
            if m.get("role") == "assistant":
                content = str(m.get("content", "") or "")
                content_clean = strip_think_block(content)
                if content_clean and len(content_clean) > 50 and not (m.get("tool_calls") or []):
                    sources.append(content_clean[:2000])
        if sources:
            pairs.append((q_key, "\n\n".join(sources)))

    random.shuffle(pairs)
    pairs = pairs[:num]

    results = []
    for i, (question, sources_text) in enumerate(pairs):
        message = f"""
Question: {question}

Collected Research Sources:
{sources_text}
"""
        try:
            result = await Runner.run(agent, message, max_turns=1)
            results.append({
                "id": f"aug_answer_{i:03d}",
                "phase": "answer",
                "system": EVAL_ANSWER_PROMPT,
                "user": message,
                "assistant": result.final_output,
            })
            if (i + 1) % 20 == 0:
                print(f"  Answer: {i + 1}/{len(pairs)}")
        except Exception as e:
            print(f"  Answer {i} failed: {e}")

    _save_jsonl(results, output_path)
    print(f"Generated {len(results)} answer examples")
    return results


# ---------------------------------------------------------------------------
# 4. Observation-Reuse Augmentation
# ---------------------------------------------------------------------------

async def augment_search_with_reuse(
    search_trajs: list[dict],
    output_path: Path,
    num: int = 40,
) -> list[dict]:
    """Generate alternative reasoning paths over existing tool responses.

    For each trajectory, we keep the system prompt, user message, and ALL tool
    responses fixed. We let the teacher model regenerate the assistant messages.
    """
    agent = Agent(
        name="SearchAugmenter",
        instructions=EVAL_SYSTEM_PROMPT,
        model=create_model(),
        tools=[],  # no tools needed — we inject observations manually
    )

    trajs_with_tools = [t for t in search_trajs if t["n_tool_responses"] >= 2]
    random.shuffle(trajs_with_tools)
    trajs_with_tools = trajs_with_tools[:num]

    results = []
    for i, traj in enumerate(trajs_with_tools):
        msgs = traj["messages"]
        # Build a "guided replay" prompt: include the original user message
        # and all observations, ask the teacher to reason through them
        user_msg = ""
        observations = []
        tools_used = []
        for m in msgs:
            if m.get("role") == "user" and not user_msg:
                user_msg = str(m.get("content", ""))
            elif m.get("role") == "tool":
                observations.append(str(m.get("content", ""))[:MAX_TOOL_RESPONSE_CHARS])
            elif m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    tools_used.append(f'{fn.get("name", "")}({json.dumps(fn.get("arguments", ""))[:100]})')

        if not observations or not user_msg:
            continue

        replay_prompt = f"""{user_msg}

Below are the tool results already collected. Based on these results, provide your analysis and findings.

{chr(10).join(f'--- Tool Result {j+1} ({tools_used[j] if j < len(tools_used) else "unknown"}) ---{chr(10)}{obs}' for j, obs in enumerate(observations))}

Analyze all the above tool results and report the key findings for this research objective.
"""
        try:
            result = await Runner.run(agent, replay_prompt, max_turns=1)
            # Store as a simplified trajectory (no tool calls, just analysis)
            results.append({
                "id": f"aug_search_reuse_{i:03d}",
                "phase": "search_reuse",
                "system": EVAL_SYSTEM_PROMPT,
                "user": replay_prompt,
                "assistant": result.final_output,
            })
            if (i + 1) % 10 == 0:
                print(f"  Search reuse: {i + 1}/{len(trajs_with_tools)}")
        except Exception as e:
            print(f"  Search reuse {i} failed: {e}")

    _save_jsonl(results, output_path)
    print(f"Generated {len(results)} search-reuse examples")
    return results


MAX_TOOL_RESPONSE_CHARS = 4000


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing search trajectories for answer generation and reuse
    search_trajs = []
    for source_dir in [args.weave_dir, args.teacher_dir]:
        search_path = source_dir / "search_trajectories.jsonl"
        if search_path.exists():
            search_trajs.extend(_load_jsonl(search_path))
    print(f"Loaded {len(search_trajs)} search trajectories for augmentation")

    # 1. Intent generation
    print("\n=== Generating synthetic intents ===")
    questions = collect_questions(args.num_intent)
    intents = await generate_intents(questions, output_dir / "intent_examples.jsonl")

    # 2. Planning generation
    print("\n=== Generating synthetic plans ===")
    await generate_plans(intents[:args.num_intent], output_dir / "planning_examples.jsonl")

    # 3. Answer generation (needs search trajectories)
    if search_trajs:
        print("\n=== Generating synthetic answers ===")
        await generate_answers(search_trajs, output_dir / "answer_examples.jsonl", num=args.num_answer)

        # 4. Observation reuse
        print("\n=== Generating search-reuse examples ===")
        await augment_search_with_reuse(search_trajs, output_dir / "search_reuse_examples.jsonl", num=args.num_reuse)
    else:
        print("\nSkipping answer/reuse augmentation: no search trajectories available")


def main():
    parser = argparse.ArgumentParser(description="Synthetic data augmentation")
    parser.add_argument("--num-intent", type=int, default=150)
    parser.add_argument("--num-answer", type=int, default=100)
    parser.add_argument("--num-reuse", type=int, default=40)
    parser.add_argument("--weave-dir", type=Path, default=Path("training/data/weave_extracted"))
    parser.add_argument("--teacher-dir", type=Path, default=Path("training/data/teacher_collected"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/augmented"))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
