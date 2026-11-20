"""Specialized Agent 1: Research Agent (HybridSearchEngine + KnowledgeGraph)."""
from __future__ import annotations

from nourai.algorithm_hub import HybridSearchEngine
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from .base import AgentOutput, AgentRoleSpec, AgentTaskRequest, BaseSpecializedAgent


class ResearchSpecialistAgent(BaseSpecializedAgent):
    spec = AgentRoleSpec(
        name="research_agent",
        role="research",
        description="Retrieves evidence across BM25, Semantic Memory, and KnowledgeGraph; separates verified facts from hypotheses.",
        allowed_permissions=("knowledge:read", "doc:analyze"),
        timeout_seconds=6.0,
    )

    def __init__(self, *, hybrid_search: HybridSearchEngine, knowledge_graph: KnowledgeGraph, software_intel=None):
        self.hybrid_search = hybrid_search
        self.kg = knowledge_graph
        self.software_intel = software_intel

    def run(self, request: AgentTaskRequest) -> AgentOutput:
        res = self.hybrid_search.search(
            request.user_id,
            request.objective,
            documents=request.documents or None,
            top_k=5,
        )
        hits = list(res.get("results", []))
        if self.software_intel is not None and (request.params.get("project_root") or request.params.get("sw_op")):
            root = self.software_intel.resolve_project_root(request.params.get("project_root"))
            proj = self.software_intel.project_understanding.analyze(root, max_files=200)
            deps = self.software_intel.dependency_analyzer.analyze(root, max_files=200)
            proj_summary = (
                f"Project has {proj['total_files_scanned']} files across layers {sorted(proj['architecture_layers'].keys())}, "
                f"{deps['external_dependencies_count']} external dependencies, circular={deps['has_circular_dependencies']}."
            )
            hits.append({"text": proj_summary, "sources": ["software_intel"], "rrf_score": 0.9})
        facts = [
            {"claim": h["text"], "sources": h.get("sources", []), "rrf_score": h.get("rrf_score", 0.0)}
            for h in hits
        ]
        hypotheses = []
        if not facts:
            hypotheses.append(
                {
                    "statement": f"No local evidence found for '{request.objective[:80]}'; requires additional sources.",
                    "status": "unverified_hypothesis",
                }
            )
        summary = " ".join(f["claim"].rstrip(".") + "." for f in facts[:2]) if facts else "No supporting local evidence found."
        conf = float(res.get("verification", {}).get("confidence", 0.25 if not facts else 0.8))
        return AgentOutput(
            agent_name=self.spec.name,
            ok=True,
            summary=summary,
            facts=facts,
            hypotheses=hypotheses,
            evidence=hits,
            confidence=round(conf, 4),
            metadata={"verification": res.get("verification", {})},
        )
