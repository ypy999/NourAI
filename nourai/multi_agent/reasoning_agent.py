"""Specialized Agent 2: Reasoning Agent (AdvancedReasoningOrchestrator v3.4.1)."""
from __future__ import annotations

from nourai.reasoning_v34 import AdvancedReasoningOrchestrator
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent


class ReasoningSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="reasoning_agent",
        role="reasoning",
        description="Performs multi-stage causal analysis, hypothesis evaluation, and uncertainty calibration.",
        allowed_permissions=("knowledge:read",),
        timeout_seconds=6.0,
    )

    def __init__(self, *, reasoning_orchestrator: AdvancedReasoningOrchestrator):
        self.orchestrator = reasoning_orchestrator

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        extra_docs = list(request.documents) + [
            str(e.get("text") or e.get("content") or "") for e in request.shared_evidence
        ]
        res = self.orchestrator.reason(
            request.user_id,
            request.objective,
            session_key=request.session_key,
            documents=[d for d in extra_docs if d],
            latency_budget_ms=float(request.params.get("latency_budget_ms", 400.0)),
        )
        supported_facts = [
            {"claim": h["statement"], "origin": h["origin"], "support_score": h["support_score"]}
            for h in res.get("hypotheses", {}).get("hypotheses", [])
            if h["status"] == "SUPPORTED"
        ]
        unverified_hyps = [
            h for h in res.get("hypotheses", {}).get("hypotheses", []) if h["status"] != "SUPPORTED"
        ]
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=res.get("final_answer", ""),
            facts=supported_facts,
            hypotheses=unverified_hyps,
            evidence=request.shared_evidence,
            confidence=float(res.get("uncertainty", {}).get("calibrated_confidence", 0.5)),
            metadata={
                "causal_edges": res.get("causal_analysis", {}).get("edge_count", 0),
                "uncertainty": res.get("uncertainty", {}),
            },
        )
