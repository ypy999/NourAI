"""JABRIX AI v3.5.0 — Planning Specialist Agent (ConstraintPlanningEngine + Human Approval Gate)."""
from __future__ import annotations

import re
import time
from typing import Any

from nourai.reasoning_v34.planning_engine import ConstraintPlanningEngine
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent

_CONSEQUENTIAL_ACTION_RE = re.compile(
    r"\b(delete|drop|remove|rm\s+-rf|deploy|publish|push\s+to\s+remote|send\s+email|"
    r"transfer\s+money|pay|purchase|overwrite\s+file|write\s+file|execute\s+shell|"
    r"supprimer|publier|déployer|envoyer)\b|"
    r"(?:احذف|امسح|انشر|أرسل|ادفع|تعديل\s+ملف)",
    re.IGNORECASE,
)


class PlanningSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="planning_agent",
        role="Goal Decomposition & Constraint Planner",
        description="Decomposes complex goals into DAG steps, enforces resource constraints, and flags consequential external actions for human approval.",
        allowed_permissions=("plan:decompose", "plan:evaluate"),
        timeout_seconds=3.0,
        parallel_safe=True,
    )

    def __init__(self, planner: ConstraintPlanningEngine | None = None) -> None:
        self.planner = planner or ConstraintPlanningEngine()

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        t0 = time.perf_counter()
        constraints = dict(request.params.get("constraints", {}))
        plan = self.planner.decompose_and_plan(request.objective, constraints=constraints)

        approved = bool(request.params.get("approved", False))
        consequential = bool(_CONSEQUENTIAL_ACTION_RE.search(request.objective or ""))
        requires_approval = consequential and not approved

        facts: list[dict[str, Any]] = [
            {
                "claim": f"Decomposed into {len(plan['steps'])} DAG step(s); dag_valid={plan['evaluation']['dag_valid']}.",
                "source": "planning_agent",
                "evidence_backed": True,
            }
        ]
        hypotheses: list[dict[str, Any]] = []
        if requires_approval:
            hypotheses.append(
                {
                    "statement": "Objective requests a consequential external or destructive action; human approval is required before execution.",
                    "status": "AWAITING_HUMAN_APPROVAL",
                    "source": "planning_agent",
                }
            )

        summary = (
            f"Generated {len(plan['steps'])}-step plan (score={plan['evaluation']['plan_score']})."
            + (" Human approval required for consequential action." if requires_approval else "")
        )
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=summary,
            facts=facts,
            hypotheses=hypotheses,
            evidence=[],
            confidence=round(float(plan["evaluation"]["plan_score"]), 4),
            requires_approval=requires_approval,
            metadata={"plan": plan, "consequential_action_detected": consequential},
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
