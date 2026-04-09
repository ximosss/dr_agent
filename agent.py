import os
import json
import asyncio
import re

from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
import weave

from models import ResearchPlan, SearchObjective
from tools import fetch_webpage, local_docs_lookup, paper_search, web_search
from prompt import (
    SYSTEM_PROMPT,
    INTENT_CLARIFICATION_PROMPT,
    PLANNING_PROMPT,
    POLISH_PROMPT,
)

set_tracing_disabled(disabled=True)

weave.init(os.getenv("WEAVE_PROJECT"))


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


async def get_local_context(local_files_path: str, question: str) -> Optional[str]:
    """Use local_docs_lookup tool to get context from local files."""
    if not local_files_path:
        return None

    from tools import run_local_docs_lookup

    result = run_local_docs_lookup(
        local_files_path=local_files_path,
        question=question,
    )
    return result if result else None


async def clarify_intent(
    question: str,
    local_context: Optional[str] = None
) -> dict:
    """
    Phase 1: Intent clarification with user alignment.
    Returns structured understanding of research intent.
    """
    intent_agent = Agent(
        name="IntentClarifier",
        instructions=INTENT_CLARIFICATION_PROMPT,
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


async def create_search_plan(clarified_intent: dict) -> ResearchPlan:
    """
    Phase 2: Create detailed search plan based on clarified intent.
    Returns a ResearchPlan with objectives.
    """
    planning_agent = Agent(
        name="ResearchPlanner",
        instructions=PLANNING_PROMPT,
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

    result = await Runner.run(planning_agent, message)
    output = strip_think_block(result.final_output)

    try:
        json_start = output.find('[')
        json_end = output.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            objectives_json = json.loads(output[json_start:json_end])
        else:
            objectives_json = []
    except json.JSONDecodeError:
        objectives_json = []

    plan = ResearchPlan(
        user_question=clarified_intent['question'],
        research_type="long-form" if "report" in clarified_intent['clarified_intent'].lower() else "short-form",
        local_context_summary=clarified_intent.get('local_context'),
    )

    for obj_data in objectives_json:
        plan.objectives.append(SearchObjective(
            objective_id=obj_data.get('objective_id', len(plan.objectives) + 1),
            description=obj_data.get('description', ''),
            search_type=obj_data.get('search_type', 'web'),
            mode=obj_data.get('mode'),
            priority=obj_data.get('priority', 'medium'),
            status='pending',
            keywords=obj_data.get('keywords', []),
        ))

    # If no objectives parsed, create default ones
    if not plan.objectives:
        plan.objectives = [
            SearchObjective(
                objective_id=1,
                description="Search web for general information",
                search_type="web",
                priority="high",
                keywords=[clarified_intent['question']],
            ),
            SearchObjective(
                objective_id=2,
                description="Search academic papers for authoritative sources",
                search_type="paper",
                mode="broad",
                priority="medium",
                keywords=[clarified_intent['question']],
            ),
        ]

    return plan


async def execute_search_plan(plan: ResearchPlan) -> ResearchPlan:
    """
    Phase 3: Agent loop - Execute search plan with context offloading.
    Main agent executes objectives, updates plan after each tool call.
    """
    main_agent = Agent(
        name="DeepResearchAgent",
        instructions=SYSTEM_PROMPT,
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

        # Build task message for this objective
        task_message = f"""
Current Research Plan:
{plan.to_context_string()}

Your current task is Objective #{current_objective.objective_id}:
- Description: {current_objective.description}
- Search type: {current_objective.search_type}
- Mode: {current_objective.mode or 'N/A'}
- Suggested keywords: {', '.join(current_objective.keywords)}

Execute this search objective using the appropriate tool. After getting results,
provide a brief summary of what you found that's relevant to the research question.
"""

        result = await Runner.run(main_agent, task_message, max_turns=30)
        output = strip_think_block(result.final_output)

        # Update plan with results
        plan.mark_completed(current_objective.objective_id, output[:500])
        plan.collected_sources.append({
            "objective_id": current_objective.objective_id,
            "summary": output,
        })

        print(f"[Completed] Objective #{current_objective.objective_id}")

        # User checkpoint: allow intervention
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
    return strip_think_block(result.final_output)


async def human_in_the_loop() -> tuple[str, Optional[str]]:
    """
    Pre-loop: Collect user question and optional local files.
    """
    print("\n" + "="*60)
    print("Deep Research Agent")
    print("="*60)

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


async def main():
    # init_observability()

    # Pre-loop: Human in the loop - collect question and context
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

    # Phase 1: Intent clarification
    print("\n[Phase 1] Clarifying research intent...")
    clarified_intent = await clarify_intent(question, local_context)

    print("\n" + "-"*40)
    print("Clarified Intent:")
    print(clarified_intent['clarified_intent'][:500])
    print("-"*40)

    confirm = input("\nDoes this capture your intent? (y/n/edit): ").strip().lower()
    if confirm == 'n':
        print("Exiting. Please refine your question and try again.")
        return
    elif confirm == 'edit':
        new_intent = input("Enter clarification: ").strip()
        clarified_intent['clarified_intent'] += f"\n\nUser clarification: {new_intent}"

    # Phase 2: Search planning
    print("\n[Phase 2] Creating search plan...")
    plan = await create_search_plan(clarified_intent)

    print("\n" + "-"*40)
    print("Research Plan:")
    print(plan.to_context_string())
    print("-"*40)

    confirm = input("\nProceed with this plan? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Exiting. Please try again with a different approach.")
        return

    # Phase 3: Execute search plan (Agent loop)
    print("\n[Phase 3] Executing search plan...")
    plan = await execute_search_plan(plan)

    # Phase 4: Generate final report
    print("\n[Phase 4] Generating final report...")
    final_report = await generate_final_report(plan)

    print("\n" + "="*60)
    print("RESEARCH REPORT")
    print("="*60)
    print(final_report)
    print("="*60)
    print("\n[End] Thank you for using Deep Research Agent!")


if __name__ == "__main__":
    asyncio.run(main())
