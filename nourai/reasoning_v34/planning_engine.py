"""JABRIX AI v3.4.0 — Constraint-Aware Goal Decomposition & Plan Evaluation Engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


@dataclass
class ReasoningStep:
    step_id: str
    stage: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    estimated_ms: float = 15.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConstraintPlanningEngine:
    """Decomposes complex goals into a constraint-validated, dependency-ordered reasoning plan."""

    def decompose_and_plan(
        self,
        goal: str,
        *,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cons = dict(constraints or {})
        max_steps = max(2, min(12, int(cons.get("max_steps", 8))))
        latency_budget_ms = float(cons.get("latency_budget_ms", 500.0))
        ram_budget_mb = float(cons.get("ram_budget_mb", 2048.0))

        # Split compound goals into sub-goals
        sub_goals = [
            p.strip()
            for p in re.split(r"\s+(?:and then|then|and|ثم|و|puis|et)\s+|[;]", (goal or "").strip(), flags=re.I)
            if len(p.strip()) >= 4
        ]
        if not sub_goals:
            sub_goals = [(goal or "").strip()]

        steps: list[ReasoningStep] = [
            ReasoningStep("s1_retrieve", "hybrid_retrieval", f"Retrieve hybrid evidence for: {sub_goals[0][:80]}", [], 20.0),
            ReasoningStep("s2_causal", "causal_analysis", "Extract cause-and-effect relationships from query and evidence", ["s1_retrieve"], 10.0),
            ReasoningStep("s3_hypothesize", "hypothesis_evaluation", "Generate and test candidate hypotheses against evidence", ["s1_retrieve", "s2_causal"], 15.0),
            ReasoningStep("s4_verify", "self_verification", "Extract claims, match evidence, detect contradictions and hallucinations", ["s3_hypothesize"], 15.0),
            ReasoningStep("s5_uncertainty", "uncertainty_calibration", "Calibrate confidence and identify missing evidence", ["s4_verify"], 5.0),
        ]

        # Under very tight latency budget (<45 ms), prune optional causal step to stay within budget
        pruned_steps: list[str] = []
        if latency_budget_ms < 45.0:
            steps = [
                ReasoningStep("s1_retrieve", "hybrid_retrieval", f"Fast hybrid retrieval for: {sub_goals[0][:80]}", [], 15.0),
                ReasoningStep("s4_verify", "self_verification", "Fast claim & contradiction verification", ["s1_retrieve"], 10.0),
                ReasoningStep("s5_uncertainty", "uncertainty_calibration", "Calibrate confidence", ["s4_verify"], 5.0),
            ]
            pruned_steps.append("s2_causal")
            pruned_steps.append("s3_hypothesize")

        steps = steps[:max_steps]
        evaluation = self.evaluate_plan(steps, latency_budget_ms=latency_budget_ms, ram_budget_mb=ram_budget_mb)
        return {
            "goal": goal,
            "sub_goals": sub_goals[:6],
            "steps": [s.to_dict() for s in steps],
            "pruned_steps": pruned_steps,
            "constraints": {
                "max_steps": max_steps,
                "latency_budget_ms": latency_budget_ms,
                "ram_budget_mb": ram_budget_mb,
            },
            "evaluation": evaluation,
        }

    @staticmethod
    def evaluate_plan(
        steps: list[ReasoningStep],
        *,
        latency_budget_ms: float = 500.0,
        ram_budget_mb: float = 2048.0,
    ) -> dict[str, Any]:
        violations: list[str] = []
        seen: set[str] = set()
        for st in steps:
            for dep in st.depends_on:
                if dep not in seen:
                    violations.append(f"unresolved_or_cyclic_dependency:{st.step_id}->{dep}")
            seen.add(st.step_id)

        est_ms = round(sum(s.estimated_ms for s in steps), 2)
        if est_ms > latency_budget_ms:
            violations.append(f"estimated_latency_{est_ms}ms_exceeds_budget_{latency_budget_ms}ms")
        if ram_budget_mb < 128.0:
            violations.append("ram_budget_below_safe_minimum_128mb")

        plan_score = round(max(0.0, 1.0 - 0.25 * len(violations)), 3)
        return {
            "dag_valid": len([v for v in violations if "dependency" in v]) == 0,
            "estimated_total_ms": est_ms,
            "constraint_violations": violations,
            "plan_score": plan_score,
        }
