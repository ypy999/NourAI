"""JABRIX AI v3.5.0 — Verification Specialist Agent (SelfVerificationEngine + UncertaintyEngine)."""
from __future__ import annotations

import time
from typing import Any

from nourai.reasoning_v34.self_verification import SelfVerificationEngine
from nourai.reasoning_v34.uncertainty_engine import UncertaintyEngine
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent


class VerificationSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="verification_agent",
        role="Fact Verification & Anti-Hallucination Auditor",
        description="Verifies candidate claims against shared evidence, catches contradictions, and demotes unsupported claims to hypotheses.",
        allowed_permissions=("verify:claims", "verify:uncertainty"),
        timeout_seconds=4.0,
        parallel_safe=True,
    )

    def __init__(
        self,
        verifier: SelfVerificationEngine | None = None,
        uncertainty: UncertaintyEngine | None = None,
        software_intel=None,
    ) -> None:
        self.verifier = verifier or SelfVerificationEngine()
        self.uncertainty = uncertainty or UncertaintyEngine()
        self.software_intel = software_intel

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        t0 = time.perf_counter()
        evidence = list(request.shared_evidence) + [
            {"text": d, "source": "document"} for d in request.documents if d
        ]
        candidate_claims: list[str] = list(request.params.get("candidate_claims", []))
        candidate_text = str(request.params.get("candidate_text", "")).strip()
        if not candidate_text and candidate_claims:
            candidate_text = ". ".join(candidate_claims)
        if not candidate_text:
            candidate_text = request.objective

        v_report = self.verifier.verify_response(candidate_text, evidence)
        u_report = self.uncertainty.estimate(
            sub_goals=list(request.params.get("sub_goals", [request.objective])),
            evidence=evidence,
            hypothesis_report={"contested_count": len(v_report.get("contradictions", []))},
            verification_report=v_report,
        )

        verified_facts: list[dict[str, Any]] = [
            {
                "claim": item["claim"],
                "source": "verification_agent",
                "evidence_backed": True,
                "overlap": item.get("overlap", 0.0),
            }
            for item in v_report.get("supported_claims", [])
        ]
        demoted_hypotheses: list[dict[str, Any]] = [
            {
                "statement": item["claim"],
                "status": "CONTRADICTED" if item.get("reason") in ("factual_contradiction", "polarity_contradiction") else "UNSUPPORTED_HYPOTHESIS",
                "reason": item.get("reason", "missing_evidence"),
                "source": "verification_agent",
            }
            for item in v_report.get("unsupported_claims", [])
        ]

        guard_report = None
        patch_check = request.params.get("proposed_patch")
        if isinstance(patch_check, dict) and self.software_intel is not None:
            guard_report = self.software_intel.regression_guard.check_patch(
                str(patch_check.get("target_file", "")),
                str(patch_check.get("original_content", "")),
                str(patch_check.get("proposed_content", "")),
            )
            if not guard_report["passed"]:
                for brk in guard_report.get("breaking_changes", []):
                    demoted_hypotheses.append(
                        {"statement": brk, "status": "BREAKING_API_CHANGE", "source": "regression_guard"}
                    )
                for rm_t in guard_report.get("removed_tests", []):
                    demoted_hypotheses.append(
                        {"statement": f"Removed test: {rm_t}", "status": "REMOVED_TEST", "source": "regression_guard"}
                    )

        summary = (
            f"Verified {len(verified_facts)}/{v_report.get('claims_checked', 0)} claim(s); "
            f"demoted {len(demoted_hypotheses)} unsupported/contradicted claim(s); "
            f"contradictions={len(v_report.get('contradictions', []))}."
        )
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=summary,
            facts=verified_facts,
            hypotheses=demoted_hypotheses,
            evidence=evidence[:10],
            confidence=round(float(u_report["calibrated_confidence"]), 4),
            metadata={"verification": v_report, "uncertainty": u_report, "regression_guard": guard_report},
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )
