"""B. Reasoning Engine — Evidence-based multi-step analysis, alternative evaluation, and uncertainty detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .perception import PerceptionOutput


@dataclass
class AlternativeOption:
    strategy: str
    tools: list[str]
    score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningOutput:
    analysis_steps: list[str]
    evidence_count: int
    alternatives: list[AlternativeOption]
    selected_strategy: str
    uncertainty_score: float  # 0.0 (certain) .. 1.0 (highly uncertain)
    uncertainty_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.8
    # v3.2 Advanced Reasoning fields (backward-compatible defaults)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    math_verification: dict[str, Any] | None = None
    planning_quality_score: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["alternatives"] = [a.to_dict() for a in self.alternatives]
        return d


class ReasoningEngine:
    """Explicit evidence-based reasoning and alternative strategy evaluator."""

    def analyze(
        self,
        perception: PerceptionOutput,
        *,
        retrieved_memories: list[dict[str, Any]],
        graph_relations: list[dict[str, Any]],
        available_tools: list[str],
    ) -> ReasoningOutput:
        steps: list[str] = [
            f"Detected language='{perception.language}', intent='{perception.primary_intent}', type='{perception.instruction_type}'.",
            f"Context retrieved: {len(retrieved_memories)} memory items, {len(graph_relations)} knowledge graph relations.",
        ]
        evidence_count = len(retrieved_memories) + len(graph_relations)
        alternatives: list[AlternativeOption] = []

        # Candidate 1: Specialized tool execution
        if perception.primary_intent == "calculation" and "python_sandbox" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="sandbox_computation",
                    tools=["python_sandbox"],
                    score=0.96,
                    rationale="Deterministic AST-verified arithmetic avoids neural hallucination.",
                )
            )
        if perception.primary_intent == "file_read" and "file_read" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="workspace_file_inspection",
                    tools=["file_read", "document_analysis"],
                    score=0.92,
                    rationale="Direct workspace file read + statistical analysis.",
                )
            )
        if perception.primary_intent == "file_write" and "file_write" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="approved_file_write",
                    tools=["file_write"],
                    score=0.90,
                    rationale="Requires explicit human approval and creates rollback snapshot.",
                )
            )
        if perception.primary_intent == "document_analysis" and "document_analysis" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="document_statistical_analysis",
                    tools=["document_analysis"],
                    score=0.93,
                    rationale="Extracts language, counts, and top keywords safely.",
                )
            )
        if perception.primary_intent == "database_query" and "database_query" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="readonly_sql_query",
                    tools=["database_query"],
                    score=0.94,
                    rationale="Read-only SELECT query with strict DML/DDL prohibition.",
                )
            )
        if perception.primary_intent == "report_generation" and "report_generation" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="knowledge_and_report_pipeline",
                    tools=["knowledge_retrieval", "report_generation"],
                    score=0.91,
                    rationale="Combines retrieved evidence with structured Markdown report generation.",
                )
            )
        if perception.primary_intent == "web_search" and "web_search" in available_tools:
            alternatives.append(
                AlternativeOption(
                    strategy="live_web_retrieval",
                    tools=["web_search"],
                    score=0.85,
                    rationale="SSRF-protected live web retrieval for current external facts.",
                )
            )

        # Always include knowledge retrieval + neural synthesis as a scored alternative
        kb_score = 0.88 if evidence_count > 0 else 0.62
        alternatives.append(
            AlternativeOption(
                strategy="knowledge_grounded_response",
                tools=["knowledge_retrieval"] if "knowledge_retrieval" in available_tools else [],
                score=kb_score,
                rationale="Grounds answer in persistent 4-tier memory and Knowledge Graph.",
            )
        )

        alternatives.sort(key=lambda a: a.score, reverse=True)
        selected = alternatives[0]
        steps.append(f"Evaluated {len(alternatives)} strategies; selected '{selected.strategy}' (score={selected.score:.2f}).")

        # Uncertainty detection
        uncertainty_reasons: list[str] = []
        uncertainty = 0.1
        if evidence_count == 0 and selected.strategy == "knowledge_grounded_response":
            uncertainty += 0.45
            uncertainty_reasons.append("no_local_memory_or_graph_evidence_found")
        if perception.primary_intent == "file_read" and not perception.extracted_paths:
            uncertainty += 0.35
            uncertainty_reasons.append("missing_explicit_file_path")
        if perception.complexity_score >= 0.7:
            uncertainty += 0.15
            uncertainty_reasons.append("multi_step_high_complexity_request")
        uncertainty = round(min(1.0, uncertainty), 2)
        confidence = round(max(0.05, 1.0 - uncertainty), 2)

        # v3.2: Constraint analysis
        constraints: list[dict[str, Any]] = [
            {"type": "language", "value": perception.language, "enforced": True},
            {"type": "least_privilege", "value": "tool_permissions_required", "enforced": True},
        ]
        if perception.instruction_type == "destructive_request":
            constraints.append({"type": "human_approval", "value": "explicit_approval_required", "enforced": True})
        if perception.primary_intent == "file_read" and not perception.extracted_paths:
            constraints.append({"type": "missing_input", "value": "file_path_not_specified", "enforced": False})

        # v3.2: Hypothesis generation (falsifiable verification criteria)
        hypotheses: list[dict[str, Any]] = [
            {
                "statement": f"Strategy '{selected.strategy}' will satisfy objective with verifiable tool output.",
                "verification_criterion": f"All tools in {selected.tools or ['knowledge_retrieval']} return ok=True with non-empty verified output.",
                "prior_confidence": confidence,
            }
        ]

        # v3.2: Contradiction detection across retrieved memories
        contradictions: list[dict[str, Any]] = []
        seen_kv: dict[str, set[str]] = {}
        import re as _re
        kv_pat = _re.compile(r"([A-Za-z\u0600-\u06FF0-9 _-]{3,35})\s*(?:is|est|هو|هي|=)\s*([^\n.;]{2,50})", _re.I)
        for mem in retrieved_memories:
            m = kv_pat.search(str(mem.get("content", "")))
            if m:
                k, v = m.group(1).strip().lower(), m.group(2).strip().lower()
                seen_kv.setdefault(k, set()).add(v)
        for k, vals in seen_kv.items():
            if len(vals) > 1:
                contradictions.append({"subject": k, "conflicting_values": sorted(vals)})
        if contradictions:
            uncertainty = round(min(1.0, uncertainty + 0.2), 2)
            confidence = round(max(0.05, 1.0 - uncertainty), 2)
            uncertainty_reasons.append("contradictory_retrieved_memories")

        # v3.2: Multi-step math pre-verification (safe AST check)
        math_verification: dict[str, Any] | None = None
        if perception.extracted_math:
            math_verification = {
                "expression": perception.extracted_math,
                "method": "ast_sandbox_required",
                "step_count": max(1, sum(perception.extracted_math.count(op) for op in ("+", "-", "*", "/"))),
            }

        planning_quality = round(max(0.1, min(1.0, selected.score - (0.15 if contradictions else 0.0))), 3)

        return ReasoningOutput(
            analysis_steps=steps,
            evidence_count=evidence_count,
            alternatives=alternatives,
            selected_strategy=selected.strategy,
            uncertainty_score=uncertainty,
            uncertainty_reasons=uncertainty_reasons,
            confidence=confidence,
            constraints=constraints,
            hypotheses=hypotheses,
            contradictions=contradictions,
            math_verification=math_verification,
            planning_quality_score=planning_quality,
        )
