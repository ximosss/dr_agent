import os
import json
import asyncio
import re
from datetime import datetime
from pathlib import Path

from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
import weave

from models import ResearchPlan, SearchObjective
from tools import clear_fetch_cache, fetch_webpage, local_docs_lookup, paper_search, web_search, run_local_docs_lookup
from prompt import (
    SYSTEM_PROMPT,
    INTENT_CLARIFICATION_PROMPT,
    PLANNING_PROMPT,
    POLISH_PROMPT,
    EVAL_SYSTEM_PROMPT,
    EVAL_INTENT_CLARIFICATION_PROMPT,
    EVAL_PLANNING_PROMPT,
    EVAL_ANSWER_PROMPT,
)

from evals import extract_final_answer, load_benchmark, score_prediction
from training.benchmark_splits import filter_examples_by_partition

# Agent flow tracking

set_tracing_disabled(disabled=True)
weave.init(os.getenv("WANDB_PROJECT"))


OBJECTIVE_RESULT_MAX_CHARS = 2000

# Helper functions

def create_model():
    """Create the LiteLLM model for agents."""
    return LitellmModel(
        model="hosted_vllm/" + os.getenv("MODEL_NAME_AT_ENDPOINT"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("BASE_KEY")
    )


def strip_think_block(text: str) -> str:
    pattern = r"<think>[\s\S]*?</think>\s*"
    return re.sub(pattern, "", text)


def _slugify(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "report"
    return cleaned[:max_len].rstrip("_")


async def get_local_context(local_files_path: str, question: str) -> Optional[str]:
    """Use local_docs_lookup tool to get context from local files."""
    if not local_files_path:
        return None

    result = run_local_docs_lookup(
        local_files_path=local_files_path,
        question=question,
    )
    return result if result else None


# Agent loops

async def clarify_intent(
    question: str,
    local_context: Optional[str] = None,
    eval_mode: bool = False,
) -> dict:
    """
    Phase 1: Intent clarification with user alignment.
    Returns structured understanding of research intent.
    """
    intent_agent = Agent(
        name="IntentClarifier",
        instructions=EVAL_INTENT_CLARIFICATION_PROMPT if eval_mode else INTENT_CLARIFICATION_PROMPT,
        model=create_model(),
    )

    context_info = ""
    if local_context:
        context_info = f"\n\nLocal file context:\n{local_context}"

    message = f"User question: {question}{context_info}"

    result = await Runner.run(intent_agent, message)
    output = strip_think_block(result.final_output)

    return {
        "question": question,
        "clarified_intent": output,
        "local_context": local_context,
    }


def _parse_objectives_json(text: str) -> list[dict]:
    """Best-effort JSON array extraction from LLM output."""
    # Strip think blocks, markdown fences
    text = strip_think_block(text)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip().rstrip("`")

    # Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return []


def _make_default_objectives(question: str) -> list[SearchObjective]:
    """Fallback objectives when plan parsing fails."""
    keywords = [w for w in question.split() if len(w) > 3][:5]
    return [
        SearchObjective(
            objective_id=1,
            description=f"Search the web for: {question[:120]}",
            search_type="web",
            priority="high",
            keywords=keywords or ["search"],
        ),
    ]


async def create_search_plan(
    clarified_intent: dict,
    eval_mode: bool = False,
    local_file_path: Optional[str] = None,
) -> ResearchPlan:
    """
    Phase 2: Create detailed search plan based on clarified intent.
    Returns a ResearchPlan with objectives.

    Uses plain text output + manual JSON parsing (instead of output_type)
    because vLLM endpoints may not support structured-output JSON schema.
    """
    planning_agent = Agent(
        name="ResearchPlanner",
        instructions=EVAL_PLANNING_PROMPT if eval_mode else PLANNING_PROMPT,
        model=create_model(),
    )

    message = f"""
Based on this research intent, create a search plan:

Question: {clarified_intent['question']}

Clarified Intent:
{clarified_intent['clarified_intent']}

Local Context Available: {'Yes' if clarified_intent.get('local_context') else 'No'}

Output a JSON array of search objectives.
"""

    plan = ResearchPlan(
        user_question=clarified_intent['question'],
        research_type="long-form" if "report" in clarified_intent['clarified_intent'].lower() else "short-form",
        local_context_summary=clarified_intent.get('local_context'),
        local_file_path=local_file_path,
    )

    try:
        result = await Runner.run(planning_agent, message)
        raw = strip_think_block(result.final_output)
        items = _parse_objectives_json(raw)
    except Exception as exc:
        print(f"[Warning] Planning agent failed ({exc}), using default objectives.")
        items = []

    if items:
        for obj in items:
            plan.objectives.append(
                SearchObjective(
                    objective_id=obj.get("objective_id", len(plan.objectives) + 1),
                    description=obj.get("description", ""),
                    search_type=obj.get("search_type", "web"),
                    mode=obj.get("mode"),
                    priority=obj.get("priority", "high"),
                    status="pending",
                    keywords=obj.get("keywords", []),
                )
            )
    else:
        print("[Warning] Could not parse objectives from LLM output, using defaults.")
        plan.objectives = _make_default_objectives(clarified_intent["question"])

    return plan


async def execute_search_plan(
    plan: ResearchPlan,
    allow_human: bool = True,
    eval_mode: bool = False,
) -> ResearchPlan:
    """
    Phase 3: Agent loop - Execute search plan with context offloading.

    Each objective receives the full plan context (including summaries from
    previously completed objectives) so the agent can build on earlier findings.
    """
    main_agent = Agent(
        name="DeepResearchAgent",
        instructions=EVAL_SYSTEM_PROMPT if eval_mode else SYSTEM_PROMPT,
        model=create_model(),
        tools=[web_search, fetch_webpage, paper_search, local_docs_lookup],
    )

    while not plan.all_completed():
        current_objective = plan.get_next_objective()
        if not current_objective:
            break

        current_objective.status = "in_progress"

        print(f"\n[Executing] Objective #{current_objective.objective_id}: {current_objective.description}")
        print(f"  Search type: {current_objective.search_type}, Keywords: {current_objective.keywords}")

        # Build task message with full plan context (context offloading)
        task_message = f"""
Current Research Plan (including results from completed objectives):
{plan.to_context_string()}

Your current task is Objective #{current_objective.objective_id}:
- Description: {current_objective.description}
- Search type: {current_objective.search_type}
- Mode: {current_objective.mode or 'N/A'}
- Suggested keywords: {', '.join(current_objective.keywords)}
"""

        # Inject local file path so the agent can use local_docs_lookup correctly
        if plan.local_file_path:
            task_message += f"""
A local file is available for this research task.
When using the local_docs_lookup tool, use this exact path: {plan.local_file_path}
"""

        task_message += """
Execute this search objective using the appropriate tool.
Report ONLY the raw facts you found (exact names, numbers, dates, quotes from sources).
Do NOT compute a final answer or draw conclusions — a separate agent will do that.
Build on findings from previously completed objectives if relevant.
"""

        result = await Runner.run(main_agent, task_message, max_turns=10)
        output = strip_think_block(result.final_output)

        # Store result with generous budget for context offloading
        result_summary = output[:OBJECTIVE_RESULT_MAX_CHARS]
        plan.mark_completed(current_objective.objective_id, result_summary)
        plan.collected_sources.append({
            "objective_id": current_objective.objective_id,
            "summary": output,
        })

        print(f"[Completed] Objective #{current_objective.objective_id}")

        # User checkpoint: allow intervention
        if allow_human:
            print("\n" + "-"*40)
            print(f"Progress: {sum(1 for o in plan.objectives if o.status == 'completed')}/{len(plan.objectives)} objectives completed")
            user_input = input("Press Enter to continue, or type 'skip' to skip remaining, 'add' to add objective: ").strip().lower()

            if user_input == 'skip':
                for obj in plan.objectives:
                    if obj.status == 'pending':
                        obj.status = 'completed'
                        obj.result_summary = 'Skipped by user'
            elif user_input == 'add':
                new_desc = input("Enter new objective description: ").strip()
                if new_desc:
                    new_keywords = input("Enter keywords (comma-separated): ").strip().split(',')
                    plan.objectives.append(SearchObjective(
                        objective_id=len(plan.objectives) + 1,
                        description=new_desc,
                        search_type=input("Search type (web/paper): ").strip() or "web",
                        priority="high",
                        keywords=[k.strip() for k in new_keywords],
                    ))

    return plan


async def generate_final_report(plan: ResearchPlan) -> str:
    """
    Phase 4: Polish and generate final research output.
    """
    polish_agent = Agent(
        name="ReportPolisher",
        instructions=POLISH_PROMPT,
        model=create_model(),
    )

    # Compile all collected sources
    sources_text = "\n\n".join([
        f"=== Source from Objective #{s['objective_id']} ===\n{s['summary']}"
        for s in plan.collected_sources
    ])

    message = f"""
Original Research Question: {plan.user_question}
Report Type: {plan.research_type}

Collected Research Sources:
{sources_text}

Please synthesize all the above sources into a {'comprehensive long-form report' if plan.research_type == 'long-form' else 'concise answer'}
with proper citations. Each citation should reference the objective ID it came from.
"""

    result = await Runner.run(polish_agent, message)
    final_output = strip_think_block(result.final_output)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(plan.user_question)
    output_path = results_dir / f"{slug}_{timestamp}.md"
    md_output = (
        "# Research Report\n\n"
        f"**Question:** {plan.user_question}\n\n"
        f"**Generated:** {timestamp}\n\n"
        "---\n\n"
        f"{final_output}\n"
    )
    output_path.write_text(md_output, encoding="utf-8")

    return final_output


async def generate_eval_answer(plan: ResearchPlan, question: str) -> str:
    eval_agent = Agent(
        name="EvalAnswerer",
        instructions=EVAL_ANSWER_PROMPT,
        model=create_model(),
    )

    sources_text = "\n\n".join([
        f"=== Source from Objective #{s['objective_id']} ===\n{s['summary']}"
        for s in plan.collected_sources
    ])

    message = f"""
Question: {question}

Collected Research Sources:
{sources_text}
"""

    result = await Runner.run(eval_agent, message)
    return strip_think_block(result.final_output)


async def run_autonomous_research(
    question: str,
    local_files_path: Optional[str] = None,
    eval_mode: bool = False,
) -> tuple[str, ResearchPlan]:
    local_context = None
    if local_files_path:
        local_context = await get_local_context(local_files_path, question)
        if local_context:
            print(f"  Local context loaded ({len(local_context)} chars)")

    clarified_intent = await clarify_intent(question, local_context, eval_mode=eval_mode)
    plan = await create_search_plan(
        clarified_intent,
        eval_mode=eval_mode,
        local_file_path=local_files_path,
    )

    plan = await execute_search_plan(plan, allow_human=False, eval_mode=eval_mode)

    if eval_mode:
        final_output = await generate_eval_answer(plan, question)
    else:
        final_output = await generate_final_report(plan)

    return final_output, plan

# --eval
async def run_eval(benchmark: str, num_examples: Optional[int]) -> None:
    outputs_dir = Path("eval_outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_example_path = outputs_dir / f"{benchmark}_{timestamp}.jsonl"
    summary_path = outputs_dir / f"{benchmark}_{timestamp}_results.json"

    examples = filter_examples_by_partition(
        benchmark,
        load_benchmark(benchmark),
        partition="test",
    )

    total = 0
    scored = 0
    correct = 0
    errors = 0

    print(f"[Eval] Loaded {len(examples)} examples from the reserved test split for benchmark '{benchmark}'.")
    if num_examples:
        print(f"[Eval] Limiting to first {num_examples} examples.")

    with per_example_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            if num_examples and idx >= num_examples:
                break
            total += 1
            clear_fetch_cache()
            print(f"\n[Eval] Example {idx + 1}/{num_examples or len(examples)}: {example.example_id}")
            try:
                prediction_raw, _ = await run_autonomous_research(
                    example.question,
                    local_files_path=example.file_path,
                    eval_mode=True,
                )
                prediction = extract_final_answer(prediction_raw)
                is_correct = score_prediction(prediction, example.answer)
                if is_correct is not None:
                    scored += 1
                    if is_correct:
                        correct += 1

                record = {
                    "benchmark": benchmark,
                    "example_id": example.example_id,
                    "question": example.question,
                    "prediction": prediction,
                    "prediction_raw": prediction_raw,
                    "gold": example.answer,
                    "correct": is_correct,
                    "level": example.level,
                    "file_path": example.file_path,
                    "metadata": example.metadata,
                }
            except Exception as exc:
                errors += 1
                record = {
                    "benchmark": benchmark,
                    "example_id": example.example_id,
                    "question": example.question,
                    "prediction": "",
                    "prediction_raw": "",
                    "gold": example.answer,
                    "correct": False,
                    "level": example.level,
                    "file_path": example.file_path,
                    "metadata": example.metadata,
                    "error": str(exc),
                }

            handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

    accuracy = (correct / scored) if scored else None
    summary = {
        "benchmark": benchmark,
        "partition": "test",
        "timestamp": timestamp,
        "total_examples": total,
        "scored_examples": scored,
        "correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "per_example_path": str(per_example_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print("\n[Eval] Completed.")
    print(f"[Eval] Results saved to: {summary_path}")


if __name__ == "__main__":
    print("This module provides the core agent logic.")
    print("Run the CLI via: uv run run_agent.py")
