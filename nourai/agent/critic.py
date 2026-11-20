"""E. Critic Engine — Result validation, plan evaluation, contradiction detection, and quality scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.tools.tool_engine import ToolResult
from .planning import ExecutionPlan


@dataclass
class CriticVerdict:
    passed: bool
    quality_score: float  # 0.0 .. 1.0
    plan_valid: bool
    contradictions_found: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CriticEngine:
    """Evaluates plans, verifies tool execution outputs, and checks Knowledge Graph contradictions."""

    def evaluate_plan(self, plan: ExecutionPlan, available_tools: set[str]) -> tuple[bool, list[str]]:
        issues: list[str] = []
        if not plan.steps:
            issues.append("plan has zero executable steps")
        if len(plan.steps) > plan.max_steps:
            issues.append(f"plan steps ({len(plan.steps)}) exceed max_steps ({plan.max_steps})")
        seen_ids: set[str] = set()
        for step in plan.steps:
            if step.tool_name not in available_tools:
                issues.append(f"step '{step.step_id}' references unknown tool '{step.tool_name}'")
            for dep in step.depends_on:
                if dep not in seen_ids:
                    issues.append(f"step '{step.step_id}' has unresolved or cyclic dependency '{dep}'")
            seen_ids.add(step.step_id)
        return len(issues) == 0, issues

    def verify_results(
        self,
        plan: ExecutionPlan,
        step_results: list[ToolResult],
        *,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> CriticVerdict:
        issues: list[str] = []
        ok_count = sum(1 for r in step_results if r.ok)
        total = max(1, len(step_results))

        for r in step_results:
            if not r.ok:
                issues.append(f"tool '{r.tool_name}' failed: {r.error} ({r.error_code})")
            elif not r.output:
                issues.append(f"tool '{r.tool_name}' succeeded with empty output")

        contradictions: list[dict[str, Any]] = []
        if knowledge_graph is not None:
            contradictions = knowledge_graph.detect_contradictions()
            if contradictions:
                issues.append(f"knowledge graph contains {len(contradictions)} unresolved contradiction(s)")

        base_score = ok_count / total
        penalty = min(0.5, 0.1 * len(issues))
        quality_score = round(max(0.0, min(1.0, base_score - penalty)), 3)
        passed = ok_count == len(step_results) and len(step_results) > 0
        return CriticVerdict(
            passed=passed,
            quality_score=quality_score,
            plan_valid=True,
            contradictions_found=contradictions,
            issues=issues,
        )
