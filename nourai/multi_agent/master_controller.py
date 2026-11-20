"""JABRIX AI v3.5.0 — Master Controller for Multi-Agent Cognitive Architecture.

Analyzes incoming tasks (AR/EN/FR), decomposes requests, lazily invokes only needed
specialized agents, enforces execution budgets and role permissions, coordinates
collaboration and conflict resolution, enforces anti-hallucination verification,
and gates consequential external actions behind human approval.
"""
from __future__ import annotations

import re
import time
from typing import Any

from nourai.algorithm_hub import AdaptiveCache, HybridSearchEngine, SemanticMemoryEngine
from nourai.evaluation.self_eval import SelfEvaluator
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.monitoring.hardware import HardwareMonitor
from nourai.reasoning_v34 import (
    AdvancedReasoningOrchestrator,
    ConstraintPlanningEngine,
    SelfVerificationEngine,
    UncertaintyEngine,
)
from .base import AgentOutput, AgentTaskRequest, LazyAgentRegistry
from .coding_agent import CodingSpecialistAgent
from .collaboration_engine import CollaborationEngine
from .learning_agent import LearningSpecialistAgent
from .memory_agent import MemorySpecialistAgent
from .planning_agent import PlanningSpecialistAgent
from .prediction_agent import PredictionSpecialistAgent
from .reasoning_agent import ReasoningSpecialistAgent
from .research_agent import ResearchSpecialistAgent
from .verification_agent import VerificationSpecialistAgent

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_FRENCH_RE = re.compile(r"\b(le|la|les|des|une|pour|avec|est|dans|mais|analyse|plan|vérifier|prévoir)\b", re.I)

_CODE_RE = re.compile(r"\b(calculate|compute|math|python|code|eval|formula|expression|احسب|معادلة|calculer)\b|\d+\s*[\+\-\*/]\s*\d+", re.I)
_PLAN_RE = re.compile(
    r"\b(plan|roadmap|steps|schedule|milestone|decompose|delete|drop|deploy|publish|write\s+file|send\s+email|"
    r"planifier|étapes|supprimer|déployer)\b|(?:خطة|خطوات|احذف|انشر)",
    re.I,
)
_PREDICT_RE = re.compile(r"\b(predict|forecast|trend|kalman|series|telemetry|latency\s+trend|توقع|تنبؤ|prévoir)\b", re.I)
_LEARN_RE = re.compile(r"\b(diagnose|failure|regression|self-eval|learn\s+from|audit\s+failures|تحليل\s+الأخطاء|échec)\b", re.I)
_MEMORY_RE = re.compile(r"\b(remember|recall|memory|my\s+notes|saved|تذكر|ذاكرة|mémoire|souvenir)\b", re.I)
_RESEARCH_RE = re.compile(r"\b(research|search|find|evidence|document|compare|source|graph|ابحث|بحث|مصادر|rechercher)\b", re.I)


class MasterController:
    """Central coordinator for the JABRIX AI v3.5.0 Multi-Agent Cognitive Architecture."""

    def __init__(
        self,
        *,
        hybrid_search: HybridSearchEngine | None = None,
        semantic_memory: SemanticMemoryEngine | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        reasoning_orchestrator: AdvancedReasoningOrchestrator | None = None,
        verifier: SelfVerificationEngine | None = None,
        uncertainty: UncertaintyEngine | None = None,
        self_evaluator: SelfEvaluator | None = None,
        hardware: HardwareMonitor | None = None,
        cache: AdaptiveCache | None = None,
        software_intel=None,
        project_intel=None,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.semantic_memory = semantic_memory
        self.knowledge_graph = knowledge_graph
        self.reasoning_orchestrator = reasoning_orchestrator
        self.verifier = verifier or (reasoning_orchestrator.self_verifier if reasoning_orchestrator else SelfVerificationEngine())
        self.uncertainty = uncertainty or (reasoning_orchestrator.uncertainty if reasoning_orchestrator else UncertaintyEngine())
        self.self_evaluator = self_evaluator or SelfEvaluator()
        self.hardware = hardware or HardwareMonitor()
        self.cache = cache
        self.software_intel = software_intel
        self.project_intel = project_intel

        self.registry = LazyAgentRegistry()
        self._register_all_agents_lazily()
        self.collaboration = CollaborationEngine(
            self.registry,
            verifier=self.verifier,
            hardware=self.hardware,
        )

    def _register_all_agents_lazily(self) -> None:
        self.registry.register_lazy(
            ResearchSpecialistAgent.spec,
            lambda: ResearchSpecialistAgent(
                hybrid_search=self.hybrid_search,
                knowledge_graph=self.knowledge_graph,
                software_intel=self.software_intel,
            ),
        )
        self.registry.register_lazy(
            ReasoningSpecialistAgent.spec,
            lambda: ReasoningSpecialistAgent(
                reasoning_orchestrator=self.reasoning_orchestrator,
            ),
        )
        self.registry.register_lazy(
            CodingSpecialistAgent.spec,
            lambda: CodingSpecialistAgent(
                software_intel=self.software_intel,
                project_intel=self.project_intel,
            ),
        )
        self.registry.register_lazy(
            MemorySpecialistAgent.spec,
            lambda: MemorySpecialistAgent(semantic_memory=self.semantic_memory),
        )
        self.registry.register_lazy(
            PlanningSpecialistAgent.spec,
            lambda: PlanningSpecialistAgent(planner=ConstraintPlanningEngine()),
        )
        self.registry.register_lazy(
            VerificationSpecialistAgent.spec,
            lambda: VerificationSpecialistAgent(
                verifier=self.verifier,
                uncertainty=self.uncertainty,
                software_intel=self.software_intel,
            ),
        )
        self.registry.register_lazy(
            PredictionSpecialistAgent.spec,
            lambda: PredictionSpecialistAgent(hardware=self.hardware),
        )
        self.registry.register_lazy(
            LearningSpecialistAgent.spec,
            lambda: LearningSpecialistAgent(self_evaluator=self.self_evaluator),
        )

    @staticmethod
    def detect_language(text: str, requested: str | None = None) -> str:
        if requested and requested in ("arabic", "english", "french"):
            return requested
        if _ARABIC_RE.search(text or ""):
            return "arabic"
        if _FRENCH_RE.search(text or ""):
            return "french"
        return "english"

    def analyze_and_assign(
        self,
        objective: str,
        *,
        documents: list[str] | None = None,
        params: dict[str, Any] | None = None,
        requested_agents: list[str] | None = None,
        max_agents: int = 5,
    ) -> dict[str, Any]:
        """Decompose objective and select the minimal set of specialized agents within budget."""
        p = dict(params or {})
        docs = list(documents or [])
        sub_tasks = [
            s.strip()
            for s in re.split(r"\s+(?:and then|then|and|ثم|و|puis|et)\s+|[;]", (objective or "").strip(), flags=re.I)
            if len(s.strip()) >= 3
        ] or [(objective or "").strip()]

        if requested_agents:
            selected = [a for a in requested_agents if any(c["name"] == a for c in self.registry.catalog())]
        else:
            selected = []
            if p.get("memory_op") or _MEMORY_RE.search(objective):
                selected.append("memory_agent")
            if docs or _RESEARCH_RE.search(objective) or self.hybrid_search is not None:
                selected.append("research_agent")
            if p.get("expression") or p.get("sw_op") or p.get("pi_op") or _CODE_RE.search(objective):
                selected.append("coding_agent")
            if _PLAN_RE.search(objective) or p.get("constraints"):
                selected.append("planning_agent")
            if p.get("series") or _PREDICT_RE.search(objective):
                selected.append("prediction_agent")
            if p.get("tasks") is not None or _LEARN_RE.search(objective):
                selected.append("learning_agent")
            selected.append("reasoning_agent")
            selected.append("verification_agent")

        # Deduplicate while preserving order and enforce budget
        deduped: list[str] = []
        for a in selected:
            if a not in deduped:
                deduped.append(a)
        cap = max(1, min(8, int(max_agents)))
        # Ensure verification_agent is kept if budget trims list
        if len(deduped) > cap:
            keep_verify = "verification_agent" in deduped
            deduped = deduped[: (cap - 1 if keep_verify else cap)]
            if keep_verify and "verification_agent" not in deduped:
                deduped.append("verification_agent")

        stage2 = [a for a in ("reasoning_agent", "verification_agent") if a in deduped]
        stage1 = [a for a in deduped if a not in stage2]
        if not stage1 and stage2:
            stage1 = [stage2[0]]
            stage2 = stage2[1:]

        return {
            "sub_tasks": sub_tasks[:6],
            "assigned_agents": deduped,
            "stage1_agents": stage1,
            "stage2_agents": stage2,
        }

    def execute(
        self,
        *,
        user_id: str,
        objective: str,
        language: str | None = None,
        session_key: str = "default",
        documents: list[str] | None = None,
        params: dict[str, Any] | None = None,
        requested_agents: list[str] | None = None,
        max_agents: int = 5,
        max_iterations: int = 2,
        timeout_seconds: float = 5.0,
        force_sequential: bool = False,
    ) -> dict[str, Any]:
        """Full MasterController execution pipeline with safety, verification, and budget enforcement."""
        t0 = time.perf_counter()
        lang = self.detect_language(objective, requested=language)
        p = dict(params or {})
        docs = list(documents or [])

        assignment = self.analyze_and_assign(
            objective,
            documents=docs,
            params=p,
            requested_agents=requested_agents,
            max_agents=max_agents,
        )

        req = AgentTaskRequest(
            user_id=user_id,
            objective=objective,
            language=lang,
            session_key=session_key,
            documents=docs,
            shared_evidence=[{"text": d, "source": "user_document"} for d in docs if d],
            params=p,
        )

        collab = self.collaboration.collaborate(
            stage1_agents=assignment["stage1_agents"],
            stage2_agents=assignment["stage2_agents"],
            request=req,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            force_sequential=force_sequential,
        )

        outputs: list[AgentOutput] = collab["outputs"]
        conflict_report = collab["conflict_report"]

        # Human approval gate check
        requires_approval = any(o.requires_approval for o in outputs)
        status = "AWAITING_APPROVAL" if requires_approval else "COMPLETED"

        # Strict anti-hallucination aggregation:
        # Only evidence-backed, non-rejected claims enter verified_facts; everything else is a hypothesis.
        def _norm_claim(txt: str) -> str:
            return txt.strip().rstrip(".").lower()

        verified_facts: list[dict[str, Any]] = []
        hypotheses: list[dict[str, Any]] = []
        rejected_set = {_norm_claim(c["claim"]) for c in conflict_report.get("rejected_claims", [])}

        # Check if verification_agent ran; if it flagged unsupported claims, never promote them to facts
        verification_out = next((o for o in outputs if o.agent_name == "verification_agent" and o.ok), None)
        unsupported_by_verifier = {
            _norm_claim(h["statement"]) for h in (verification_out.hypotheses if verification_out else [])
        }

        seen_facts: set[str] = set()
        for o in outputs:
            if not o.ok:
                continue
            for f in o.facts:
                claim_txt = str(f.get("claim", "")).strip()
                norm_txt = _norm_claim(claim_txt)
                if not claim_txt or norm_txt in seen_facts:
                    continue
                if norm_txt in rejected_set or norm_txt in unsupported_by_verifier or not f.get("evidence_backed", False):
                    hypotheses.append(
                        {
                            "statement": claim_txt,
                            "status": "DEMOTED_UNVERIFIED_CLAIM",
                            "source": o.agent_name,
                        }
                    )
                else:
                    seen_facts.add(norm_txt)
                    verified_facts.append(
                        {
                            "claim": claim_txt,
                            "source": o.agent_name,
                            "evidence_backed": True,
                        }
                    )
            for h in o.hypotheses:
                hypotheses.append(h)

        for rej in conflict_report.get("rejected_claims", []):
            hypotheses.append(
                {
                    "statement": rej["claim"],
                    "status": "REJECTED_IN_CONFLICT",
                    "source": rej["agent"],
                }
            )

        # Build concise final response summary
        if status == "AWAITING_APPROVAL":
            final_answer = (
                "Execution paused: consequential external or destructive action detected. "
                "Human approval is required (re-submit with approved=true after review)."
            )
        elif verified_facts:
            final_answer = " ".join(f["claim"] for f in verified_facts[:4])
        elif hypotheses:
            final_answer = (
                "Insufficient verified evidence to state confirmed facts. "
                f"Candidate hypothesis: {hypotheses[0]['statement']}"
            )
        else:
            final_answer = "Multi-agent execution completed with no factual claims generated."

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        hw_obj = self.hardware.snapshot()
        hw = hw_obj.to_dict() if hasattr(hw_obj, "to_dict") else dict(hw_obj)
        return {
            "version": "3.5.0",
            "status": status,
            "requires_approval": requires_approval,
            "user_id": user_id,
            "language": lang,
            "objective": objective,
            "final_answer": final_answer,
            "sub_tasks": assignment["sub_tasks"],
            "assigned_agents": assignment["assigned_agents"],
            "loaded_agents_count": self.registry.loaded_count(),
            "verified_facts": verified_facts,
            "hypotheses": hypotheses,
            "evidence_count": len(collab["shared_evidence"]),
            "conflicts": conflict_report["conflicts"],
            "conflict_resolutions": conflict_report["resolutions"],
            "consensus_score": conflict_report["consensus_score"],
            "iterations_executed": collab["iterations_executed"],
            "max_iterations_cap": collab["max_iterations_cap"],
            "execution_modes": collab["execution_modes"],
            "agent_outputs": [o.to_dict() for o in outputs],
            "latency_ms": elapsed_ms,
            "hardware_rss_mb": hw.get("process_rss_mb", 0.0),
        }

    def registry_status(self) -> dict[str, Any]:
        catalog = self.registry.catalog()
        return {
            "version": "3.5.0",
            "total_agents": len(catalog),
            "loaded_agents": self.registry.loaded_count(),
            "agents": catalog,
        }
