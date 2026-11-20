"""C. Planning Engine — Goal decomposition, dependency management, prioritization, and execution planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .perception import PerceptionOutput
from .reasoning import ReasoningOutput


@dataclass
class PlanStep:
    step_id: str
    description: str
    tool_name: str
    tool_input: dict[str, Any]
    priority: int = 1
    depends_on: list[str] = field(default_factory=list)
    requires_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    objective: str
    strategy: str
    steps: list[PlanStep]
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "strategy": self.strategy,
            "max_steps": self.max_steps,
            "steps": [s.to_dict() for s in self.steps],
        }


class PlanningEngine:
    """Decomposes user objectives into bounded, dependency-ordered executable steps."""

    def build_plan(
        self,
        perception: PerceptionOutput,
        reasoning: ReasoningOutput,
        *,
        max_steps: int = 8,
        explicit_params: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        params = dict(explicit_params or {})
        steps: list[PlanStep] = []
        strategy = reasoning.selected_strategy

        if strategy == "sandbox_computation" and perception.extracted_math:
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Evaluate mathematical expression in AST sandbox",
                    tool_name="python_sandbox",
                    tool_input={"expression": perception.extracted_math},
                    priority=1,
                )
            )
        elif strategy == "workspace_file_inspection":
            target_path = params.get("path") or (perception.extracted_paths[0] if perception.extracted_paths else "README.md")
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description=f"Read workspace file '{target_path}'",
                    tool_name="file_read",
                    tool_input={"path": target_path},
                    priority=1,
                )
            )
            steps.append(
                PlanStep(
                    step_id="step_2",
                    description=f"Analyze statistics and keywords of '{target_path}'",
                    tool_name="document_analysis",
                    tool_input={"path": target_path},
                    priority=2,
                    depends_on=["step_1"],
                )
            )
        elif strategy == "approved_file_write":
            target_path = params.get("path") or (perception.extracted_paths[0] if perception.extracted_paths else "reports/agent_output.txt")
            content = params.get("content") or perception.raw_request
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description=f"Write file '{target_path}' (requires explicit user approval)",
                    tool_name="file_write",
                    tool_input={"path": target_path, "content": content},
                    priority=1,
                    requires_approval=True,
                )
            )
        elif strategy == "document_statistical_analysis":
            payload: dict[str, Any] = {}
            if params.get("path") or perception.extracted_paths:
                payload["path"] = params.get("path") or perception.extracted_paths[0]
            else:
                payload["text"] = params.get("text") or perception.raw_request
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Analyze document/text metrics and language",
                    tool_name="document_analysis",
                    tool_input=payload,
                    priority=1,
                )
            )
        elif strategy == "readonly_sql_query":
            sql_match = re.search(r"(select\s+.+)", perception.raw_request, re.I)
            sql = params.get("sql") or (sql_match.group(1).strip() if sql_match else "SELECT name FROM sqlite_master WHERE type='table'")
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Execute read-only SQLite SELECT query",
                    tool_name="database_query",
                    tool_input={"sql": sql},
                    priority=1,
                )
            )
        elif strategy == "knowledge_and_report_pipeline":
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Retrieve relevant memories and Knowledge Graph facts",
                    tool_name="knowledge_retrieval",
                    tool_input={"query": perception.raw_request, "top_k": 5},
                    priority=1,
                )
            )
            steps.append(
                PlanStep(
                    step_id="step_2",
                    description="Generate structured Markdown report",
                    tool_name="report_generation",
                    tool_input={
                        "title": params.get("title") or f"NourAI Report: {perception.raw_request[:60]}",
                        "sections": [
                            {"heading": "Objective", "body": perception.raw_request},
                            {"heading": "Language", "body": perception.language},
                        ],
                        "format": "markdown",
                    },
                    priority=2,
                    depends_on=["step_1"],
                )
            )
        elif strategy == "live_web_retrieval":
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Query live web search (SSRF-protected)",
                    tool_name="web_search",
                    tool_input={"query": perception.raw_request, "language": perception.language},
                    priority=1,
                )
            )
        else:
            steps.append(
                PlanStep(
                    step_id="step_1",
                    description="Retrieve local memory and Knowledge Graph context",
                    tool_name="knowledge_retrieval",
                    tool_input={"query": perception.raw_request, "top_k": 5},
                    priority=1,
                )
            )

        # Enforce hard max_steps cap (prevents runaway plans / infinite loops)
        bounded = steps[: max(1, int(max_steps))]
        return ExecutionPlan(
            objective=perception.raw_request,
            strategy=strategy,
            steps=bounded,
            max_steps=max_steps,
        )
