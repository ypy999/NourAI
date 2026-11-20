"""JABRIX AI v3.5.0 — Learning Specialist Agent (SelfEvaluator Failure Analysis & Regression Generation)."""
from __future__ import annotations

import time
from typing import Any

from nourai.evaluation.self_eval import SelfEvaluator
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent


class LearningSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="learning_agent",
        role="Failure Diagnostics & Regression Learning Specialist",
        description="Analyzes task failures, classifies root causes, and generates human-reviewed regression tests without unverified model updates.",
        allowed_permissions=("eval:read", "eval:generate_tests"),
        timeout_seconds=4.0,
        parallel_safe=True,
    )

    def __init__(self, self_evaluator: SelfEvaluator | None = None) -> None:
        self.self_evaluator = self_evaluator or SelfEvaluator()

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        t0 = time.perf_counter()
        tasks = request.params.get("tasks")
        if isinstance(tasks, list):
            report = self.self_evaluator.analyze_tasks(tasks)
        else:
            report = self.self_evaluator.analyze_from_db(limit=int(request.params.get("limit", 100)))

        facts: list[dict[str, Any]] = [
            {
                "claim": (
                    f"Analyzed {report['total_tasks_analyzed']} task(s): "
                    f"{report['failed_count']} failed (rate={report['failure_rate']})."
                ),
                "source": "learning_agent",
                "evidence_backed": True,
            }
        ]
        hypotheses: list[dict[str, Any]] = [
            {
                "statement": f"[{rec['category']}] {rec['recommendation']}",
                "status": "IMPROVEMENT_RECOMMENDATION",
                "source": "learning_agent",
            }
            for rec in report.get("recommendations", [])[:5]
        ]

        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=(
                f"Diagnostics complete: {report['failed_count']}/{report['total_tasks_analyzed']} failures; "
                f"{len(report.get('candidate_tests', []))} candidate regression test(s) generated."
            ),
            facts=facts,
            hypotheses=hypotheses,
            evidence=[],
            confidence=0.9 if report["total_tasks_analyzed"] > 0 else 0.6,
            metadata={
                "report": report,
                "allow_auto_unverified_model_updates": False,
            },
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
