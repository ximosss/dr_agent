import os
import json
import asyncio
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, ModelSettings, RunHooks, Runner, set_tracing_disabled
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
    EVAL_JUDGE_PROMPT,
)

from evals import extract_final_answer, load_benchmark, score_prediction
from training.benchmark_splits import filter_examples_by_partition

# Agent flow tracking

set_tracing_disabled(disabled=True)
weave.init(os.getenv("WANDB_PROJECT"))


OBJECTIVE_RESULT_MAX_CHARS = 1400
COLLECTED_SOURCE_MAX_CHARS = 1200
FINAL_SOURCE_BUNDLE_MAX_CHARS = 8000


@dataclass(frozen=True)
class ToolBudget:
    limits: dict[str, int]
    max_turns: int


@dataclass
class ToolBudgetState:
    limits: dict[str, int]
    counts: dict[str, int] = field(default_factory=dict)

    def can_use(self, tool_name: str) -> bool:
        total_limit = self.limits.get("_total")
        if total_limit is not None and sum(self.counts.values()) >= total_limit:
            return False

        tool_limit = self.limits.get(tool_name)
        if tool_limit is None:
            return True
        return self.counts.get(tool_name, 0) < tool_limit

    def record_tool(self, tool_name: str) -> None:
        self.counts[tool_name] = self.counts.get(tool_name, 0) + 1

    def prompt_lines(self) -> list[str]:
        order = ("web_search", "fetch_webpage", "paper_search", "local_docs_lookup")
        lines = []
        for tool_name in order:
            limit = self.limits.get(tool_name)
            if limit is not None:
                lines.append(f"- {tool_name}: at most {limit} calls")
        total_limit = self.limits.get("_total")
        if total_limit is not None:
            lines.append(f"- total tool calls across this objective: at most {total_limit}")
        return lines

    def usage_summary(self) -> str:
        order = ("web_search", "fetch_webpage", "paper_search", "local_docs_lookup")
        parts = [f"{tool_name}={self.counts.get(tool_name, 0)}" for tool_name in order]
        parts.append(f"total={sum(self.counts.values())}")
        return ", ".join(parts)


class ToolBudgetHooks(RunHooks[ToolBudgetState]):
    async def on_tool_end(self, context, agent, tool, result) -> None:
        if isinstance(context.context, ToolBudgetState):
            context.context.record_tool(tool.name)


INTERACTIVE_TOOL_BUDGET = ToolBudget(
    limits={
        "web_search": 6,
        "fetch_webpage": 8,
        "paper_search": 4,
        "local_docs_lookup": 2,
        "_total": 18,
    },
    max_turns=18,
)
DEFAULT_EVAL_TOOL_BUDGET = ToolBudget(
    limits={
        "web_search": 5,
        "fetch_webpage": 7,
        "paper_search": 2,
        "local_docs_lookup": 1,
        "_total": 14,
    },
    max_turns=16,
)
BENCHMARK_EVAL_TOOL_BUDGETS = {
    "simpleqa": ToolBudget(
        limits={
            "web_search": 4,
            "fetch_webpage": 6,
            "paper_search": 2,
            "local_docs_lookup": 1,
            "_total": 12,
        },
        max_turns=14,
    ),
    "frames": ToolBudget(
        limits={
            "web_search": 5,
            "fetch_webpage": 7,
            "paper_search": 2,
            "local_docs_lookup": 1,
            "_total": 14,
        },
        max_turns=16,
    ),
    "gaia": ToolBudget(
        limits={
            "web_search": 6,
            "fetch_webpage": 8,
            "paper_search": 3,
            "local_docs_lookup": 1,
            "_total": 16,
        },
        max_turns=18,
    ),
}

# Helper functions

def create_model():
    """Create the LiteLLM model for agents."""
    return LitellmModel(
        model="hosted_vllm/" + os.getenv("MODEL_NAME_AT_ENDPOINT"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("BASE_KEY")
    )


def create_judge_model():
    """Create the LiteLLM model for the evaluation judge agent."""
    model_name = os.getenv("JUDGE_MODEL_NAME_AT_ENDPOINT") or os.getenv("MODEL_NAME_AT_ENDPOINT")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL")
    api_key = os.getenv("JUDGE_BASE_KEY") or os.getenv("BASE_KEY")
    return LitellmModel(
        model="hosted_vllm/" + model_name,
        base_url=base_url,
        api_key=api_key,
    )


def strip_think_block(text: str) -> str:
    pattern = r"<think>[\s\S]*?</think>\s*"
    return re.sub(pattern, "", text)


def _parse_judge_output(text: str) -> dict | None:
    text = strip_think_block(text)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.strip().rstrip("`")

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed.get("correct"), bool):
                return {
                    "correct": parsed["correct"],
                    "reason": str(parsed.get("reason", "")).strip(),
                }
        except json.JSONDecodeError:
            pass

    match = re.search(r'"correct"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if match:
        return {
            "correct": match.group(1).lower() == "true",
            "reason": "",
        }

    return None


def _slugify(text: str, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "report"
    return cleaned[:max_len].rstrip("_")


def _resolve_tool_budget(eval_mode: bool, benchmark_name: Optional[str]) -> ToolBudget:
    if not eval_mode:
        return INTERACTIVE_TOOL_BUDGET

    benchmark_key = (benchmark_name or "").strip().lower()
    return BENCHMARK_EVAL_TOOL_BUDGETS.get(benchmark_key, DEFAULT_EVAL_TOOL_BUDGET)


def _tool_enabled_checker(tool_name: str):
    def is_enabled(run_context, agent) -> bool:
        state = run_context.context
        if not isinstance(state, ToolBudgetState):
            return True
        return state.can_use(tool_name)

    return is_enabled


def _build_budgeted_tools():
    return [
        replace(web_search, is_enabled=_tool_enabled_checker("web_search")),
        replace(fetch_webpage, is_enabled=_tool_enabled_checker("fetch_webpage")),
        replace(paper_search, is_enabled=_tool_enabled_checker("paper_search")),
        replace(local_docs_lookup, is_enabled=_tool_enabled_checker("local_docs_lookup")),
    ]


def _truncate_for_context(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


def _compile_sources_text(plan: ResearchPlan) -> str:
    chunks: list[str] = []
    used = 0
    truncated = False

    for source in plan.collected_sources:
        chunk = f"=== Source from Objective #{source['objective_id']} ===\n{source['summary']}"
        if used + len(chunk) > FINAL_SOURCE_BUNDLE_MAX_CHARS:
            remaining = FINAL_SOURCE_BUNDLE_MAX_CHARS - used
            if remaining > 64:
                chunks.append(_truncate_for_context(chunk, remaining))
            truncated = True
            break
        chunks.append(chunk)
        used += len(chunk)

    if truncated:
        chunks.append("[Additional source content omitted to stay within the context budget.]")

    return "\n\n".join(chunks)


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

    return plan


async def execute_search_plan(
    plan: ResearchPlan,
    allow_human: bool = True,
    eval_mode: bool = False,
    benchmark_name: Optional[str] = None,
) -> ResearchPlan:
    """
    Phase 3: Agent loop - Execute search plan with context offloading.

    Each objective receives the full plan context (including summaries from
    previously completed objectives) so the agent can build on earlier findings.
    """
    budget = _resolve_tool_budget(eval_mode, benchmark_name)
    budget_hooks = ToolBudgetHooks()
    budgeted_tools = _build_budgeted_tools()

    main_agent = Agent(
        name="DeepResearchAgent",
        instructions=EVAL_SYSTEM_PROMPT if eval_mode else SYSTEM_PROMPT,
        model=create_model(),
        model_settings=ModelSettings(parallel_tool_calls=False),
        tools=budgeted_tools,
    )

    while not plan.all_completed():
        current_objective = plan.get_next_objective()
        if not current_objective:
            break

        current_objective.status = "in_progress"

        print(f"\n[Executing] Objective #{current_objective.objective_id}: {current_objective.description}")
        print(f"  Search type: {current_objective.search_type}, Keywords: {current_objective.keywords}")

        budget_state = ToolBudgetState(limits=dict(budget.limits))

        # Build task message with full plan context (context offloading)
        task_message = f"""
Current Research Plan (including results from completed objectives):
{plan.to_context_string()}

Your current task is Objective #{current_objective.objective_id}:
- Description: {current_objective.description}
- Search type: {current_objective.search_type}
- Mode: {current_objective.mode or 'N/A'}
- Suggested keywords: {', '.join(current_objective.keywords)}

Budget for this objective:
{chr(10).join(budget_state.prompt_lines())}
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
Stop searching as soon as you have enough evidence to answer the current objective.
"""

        try:
            result = await Runner.run(
                main_agent,
                task_message,
                context=budget_state,
                hooks=budget_hooks,
                max_turns=budget.max_turns,
            )
            output = strip_think_block(result.final_output)
        except Exception as exc:
            print(f"[Warning] Objective #{current_objective.objective_id} ended early: {exc}")
            output = f"[OBJECTIVE_ERROR] {exc}"

        # Store result with generous budget for context offloading
        result_summary = _truncate_for_context(output, OBJECTIVE_RESULT_MAX_CHARS)
        plan.mark_completed(current_objective.objective_id, result_summary)
        plan.collected_sources.append({
            "objective_id": current_objective.objective_id,
            "summary": _truncate_for_context(output, COLLECTED_SOURCE_MAX_CHARS),
            "tool_usage": budget_state.usage_summary(),
        })

        print(f"[Completed] Objective #{current_objective.objective_id}")
        print(f"  Tool usage: {budget_state.usage_summary()}")

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
    sources_text = _compile_sources_text(plan)

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

    sources_text = _compile_sources_text(plan)

    message = f"""
Question: {question}

Collected Research Sources:
{sources_text}
"""

    result = await Runner.run(eval_agent, message)
    return strip_think_block(result.final_output)


async def judge_prediction(
    question: str,
    gold: Optional[str],
    prediction: str,
    prediction_raw: str = "",
) -> dict:
    """Judge a prediction with a dedicated LLM judge, falling back to rules."""
    if gold is None:
        return {
            "correct": None,
            "method": "no_gold",
            "reason": "",
            "raw_output": "",
        }

    judge_agent = Agent(
        name="EvalJudge",
        instructions=EVAL_JUDGE_PROMPT,
        model=create_judge_model(),
    )

    message = f"""
Question:
{question}

Gold answer:
{gold}

Candidate prediction:
{prediction}

Candidate raw output:
{prediction_raw}
"""

    try:
        result = await Runner.run(judge_agent, message, max_turns=1)
        raw_output = strip_think_block(result.final_output)
        parsed = _parse_judge_output(raw_output)
        if parsed is not None:
            return {
                "correct": parsed["correct"],
                "method": "llm_judge",
                "reason": parsed.get("reason", ""),
                "raw_output": raw_output,
            }
        raise ValueError("Judge output was not parseable as the required JSON object.")
    except Exception as exc:
        fallback = score_prediction(prediction, gold)
        return {
            "correct": fallback,
            "method": "rule_fallback",
            "reason": f"Fallback after judge failure: {exc}",
            "raw_output": "",
            "error": str(exc),
        }


async def run_autonomous_research(
    question: str,
    local_files_path: Optional[str] = None,
    eval_mode: bool = False,
    benchmark_name: Optional[str] = None,
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

    plan = await execute_search_plan(
        plan,
        allow_human=False,
        eval_mode=eval_mode,
        benchmark_name=benchmark_name,
    )

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
                    benchmark_name=benchmark,
                )
                prediction = extract_final_answer(prediction_raw)
                rule_correct = score_prediction(prediction, example.answer)
                judge = await judge_prediction(
                    question=example.question,
                    gold=example.answer,
                    prediction=prediction,
                    prediction_raw=prediction_raw,
                )
                is_correct = judge["correct"]
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
                    "rule_correct": rule_correct,
                    "judge_method": judge.get("method"),
                    "judge_reason": judge.get("reason"),
                    "judge_raw_output": judge.get("raw_output"),
                    "level": example.level,
                    "file_path": example.file_path,
                    "metadata": example.metadata,
                }
                if judge.get("error"):
                    record["judge_error"] = judge["error"]
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
