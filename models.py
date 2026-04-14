"""Shared domain models for the research workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    local_file_path: Optional[str] = None
    objectives: list[SearchObjective] = field(default_factory=list)
    collected_sources: list[dict] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Convert plan to string for agent context.

        Completed objectives include their full result_summary so that
        subsequent objectives can build on earlier findings (context
        offloading rather than context isolation).
        """
        lines = [
            f"Research Question: {self.user_question}",
            f"Type: {self.research_type}",
        ]
        if self.local_file_path:
            lines.append(f"Local file path: {self.local_file_path}")
        if self.local_context_summary:
            lines.append(f"Local Context: {self.local_context_summary[:500]}...")

        lines.append("\nSearch Plan:")
        for obj in self.objectives:
            status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}[obj.status]
            lines.append(f"  {status_icon} #{obj.objective_id} [{obj.priority}] {obj.description}")
            if obj.result_summary:
                lines.append(f"      Result: {obj.result_summary}")

        return "\n".join(lines)

    def get_next_objective(self) -> Optional[SearchObjective]:
        """Get next pending objective by priority."""
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pending = [o for o in self.objectives if o.status == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda x: priority_order.get(x.priority, 1))
        return pending[0]

    def mark_completed(self, objective_id: int, result_summary: str) -> None:
        """Mark an objective as completed."""
        for obj in self.objectives:
            if obj.objective_id == objective_id:
                obj.status = "completed"
                obj.result_summary = result_summary
                break

    def all_completed(self) -> bool:
        """Check if all objectives are completed."""
        return all(o.status == "completed" for o in self.objectives)
