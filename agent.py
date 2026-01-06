import os
import json
import asyncio

from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
import weave
from utils.helpers import strip_think_block

from tools import web_search, paper_search, local_docs_lookup, summarize_sources
from prompt import (
    SYSTEM_PROMPT,
    INTENT_CLARIFICATION_PROMPT,
    PLANNING_PROMPT,
    POLISH_PROMPT,
)

os.environ["HTTP_PROXY"] = "http://localhost:8081"
os.environ["HTTPS_PROXY"] = "http://localhost:8081"

set_tracing_disabled(disabled=True)

model = os.getenv("MODEL_NAME_AT_ENDPOINT")
api_key = os.getenv("BASE_KEY")
base_url = os.getenv("BASE_URL")

weave.init("ximo_ml/deep_research_agent")


@dataclass
class SearchObjective:
    """Single search objective in the research plan."""
    objective_id: int
    description: str
    search_type: str  # "web" | "paper" | "local"
    mode: Optional[str] = None  # For papers: "precise" | "broad"
    priority: str = "medium"  # "high" | "medium" | "low"
    status: str = "pending"  # "pending" | "in_progress" | "completed"
    keywords: list[str] = field(default_factory=list)
    result_summary: Optional[str] = None


@dataclass
class ResearchPlan:
    """Context-offloaded research plan (todo list)."""
    user_question: str
    research_type: str  # "short-form" | "long-form"
    local_context_summary: Optional[str] = None
    objectives: list[SearchObjective] = field(default_factory=list)
    collected_sources: list[dict] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Convert plan to string for agent context."""
        lines = [
            f"Research Question: {self.user_question}",
            f"Type: {self.research_type}",
        ]
        if self.local_context_summary:
            lines.append(f"Local Context: {self.local_context_summary[:500]}...")

        lines.append("\nSearch Plan:")
        for obj in self.objectives:
            status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}[obj.status]
            lines.append(f"  {status_icon} #{obj.objective_id} [{obj.priority}] {obj.description}")
            if obj.result_summary:
                lines.append(f"      Result: {obj.result_summary[:200]}...")

        return "\n".join(lines)

    def get_next_objective(self) -> Optional[SearchObjective]:
        """Get next pending objective by priority."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pending = [o for o in self.objectives if o.status == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda x: priority_order.get(x.priority, 1))
        return pending[0]

    def mark_completed(self, objective_id: int, result_summary: str):
        """Mark an objective as completed."""
        for obj in self.objectives:
            if obj.objective_id == objective_id:
                obj.status = "completed"
                obj.result_summary = result_summary
                break

    def all_completed(self) -> bool:
        """Check if all objectives are completed."""
        return all(o.status == "completed" for o in self.objectives)


def create_model():
    """Create the LiteLLM model for agents."""
    return LitellmModel(
        model="hosted_vllm/" + model,
        base_url=base_url,
        api_key=api_key
    )


async def get_local_context(local_files_path: str, question: str) -> Optional[str]:
    """Use local_docs_lookup tool to get context from local files."""
    if not local_files_path:
        return None

    # Call the tool directly
    from tools import local_docs_lookup
    result = await local_docs_lookup.on_invoke_tool(
        ctx=None,
        input=json.dumps({
            "local_files_path": local_files_path,
            "question": question,
        })
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

    # Parse the intent (simplified - in production would use structured output)
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

    # Parse JSON from output
    try:
        # Try to extract JSON from the output
        json_start = output.find('[')
        json_end = output.rfind(']') + 1
        if json_start != -1 and json_end > json_start:
            objectives_json = json.loads(output[json_start:json_end])
        else:
            objectives_json = []
    except json.JSONDecodeError:
        objectives_json = []

    # Create plan
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
        tools=[web_search, paper_search, local_docs_lookup, summarize_sources],
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

        result = await Runner.run(main_agent, task_message, max_turns=300)
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

    # Refinement loop
    while True:
        print("\n[Refinement] Enter follow-up question, 'new' for new research, or 'quit' to exit:")
        user_input = input("> ").strip()

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("\nThank you for using Deep Research Agent!")
            break
        if user_input.lower() == 'new':
            return await main()

        # Handle follow-up
        print("\n[Processing follow-up...]")
        follow_up_agent = Agent(
            name="FollowUpAgent",
            instructions=SYSTEM_PROMPT,
            model=create_model(),
            tools=[web_search, paper_search, local_docs_lookup, summarize_sources],
        )

        follow_up_message = f"""
Previous research on: {question}

Previous findings summary:
{final_report[:2000]}...

User's follow-up question: {user_input}

Please answer the follow-up question, using additional searches if needed.
"""
        result = await Runner.run(follow_up_agent, follow_up_message, max_turns=30)
        output = strip_think_block(result.final_output)

        print("\n" + "="*60)
        print("FOLLOW-UP RESPONSE")
        print("="*60)
        print(output)
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
