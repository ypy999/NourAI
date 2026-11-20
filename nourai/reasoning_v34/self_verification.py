"""JABRIX AI v3.4.1 — Self-Verification Engine with Multilingual Clause Segmentation,
Negation-Aware Contradiction Detection, and Explicit Factual Contradiction vs Missing Information Separation."""
from __future__ import annotations

from typing import Any

from nourai.memory.advanced_memory import _tokenize
from .clause_segmenter import ClauseUnit, segment_clauses


class SelfVerificationEngine:
    """Audits candidate answers at atomic clause level to catch contradictions, negations, and hallucinations."""

    def __init__(self, *, support_overlap_threshold: float = 0.30):
        self.support_overlap_threshold = float(support_overlap_threshold)

    @staticmethod
    def extract_claims(text: str) -> list[str]:
        clauses = segment_clauses(text)
        return [c.text for c in clauses[:30]]

    def verify_response(
        self,
        candidate_response: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        claim_clauses = segment_clauses(candidate_response)
        if not claim_clauses:
            return {
                "claims_checked": 0,
                "supported_claims": [],
                "unsupported_claims": [],
                "contradicted_claims": [],
                "missing_info_claims": [],
                "contradictions": [],
                "clause_relations": [],
                "grounded_ratio": 0.0,
                "hallucination_risk": "HIGH",
                "verified": False,
            }

        # Segment all evidence items into atomic clauses
        ev_clauses: list[tuple[ClauseUnit, set[str], int]] = []
        ev_kv_index: dict[str, list[tuple[ClauseUnit, int]]] = {}
        for src_idx, ev in enumerate(evidence):
            txt = str(ev.get("text") or ev.get("content") or "")
            for cu in segment_clauses(txt):
                toks = set(_tokenize(cu.text))
                ev_clauses.append((cu, toks, src_idx))
                if cu.subject and cu.value:
                    ev_kv_index.setdefault(cu.subject, []).append((cu, src_idx))

        supported: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        contradicted_claims: list[dict[str, Any]] = []
        missing_info_claims: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []

        # 1. Detect cross-source contradictions among evidence clauses themselves
        for subj, entries in ev_kv_index.items():
            distinct_states = {(cu.value, cu.negated) for cu, _ in entries}
            distinct_sources = {src_i for _, src_i in entries}
            if len(distinct_states) > 1 and len(distinct_sources) > 1:
                contradictions.append(
                    {
                        "type": "evidence_conflict",
                        "subject": subj,
                        "conflicting_values": sorted(f"{'NOT ' if neg else ''}{val}" for val, neg in distinct_states),
                    }
                )

        # 2. Verify every atomic claim clause against segmented evidence
        for cu in claim_clauses:
            c_toks = set(_tokenize(cu.text))
            best_score = 0.0
            best_ev_unit: ClauseUnit | None = None
            polarity_conflict_ev: ClauseUnit | None = None
            kv_conflict_ev: ClauseUnit | None = None

            # A. Exact subject-predicate-value & polarity conflict check
            if cu.subject and cu.subject in ev_kv_index:
                for ev_u, _ in ev_kv_index[cu.subject]:
                    if cu.negated != ev_u.negated:
                        polarity_conflict_ev = ev_u
                    elif cu.value and ev_u.value and cu.value != ev_u.value:
                        # Check if values are genuinely disjoint (not a minor subset phrase)
                        v_claim_toks = set(_tokenize(cu.value))
                        v_ev_toks = set(_tokenize(ev_u.value))
                        if not (v_claim_toks and v_claim_toks <= v_ev_toks):
                            kv_conflict_ev = ev_u

            # B. Token overlap + polarity check across all evidence clauses
            for ev_u, ev_toks, _ in ev_clauses:
                if not c_toks:
                    continue
                overlap = len(c_toks & ev_toks) / len(c_toks)
                if overlap >= self.support_overlap_threshold and cu.negated != ev_u.negated:
                    # High topic overlap with opposite polarity -> factual polarity contradiction!
                    if polarity_conflict_ev is None:
                        polarity_conflict_ev = ev_u
                if overlap > best_score:
                    best_score = overlap
                    best_ev_unit = ev_u

            if kv_conflict_ev is not None:
                c_rec = {
                    "type": "claim_vs_evidence",
                    "subject": cu.subject,
                    "claim_value": cu.value,
                    "evidence_value": kv_conflict_ev.value,
                    "evidence_snippet": kv_conflict_ev.text[:160],
                }
                contradictions.append(c_rec)
                entry = {"claim": cu.text, "reason": "factual_contradiction", "best_overlap": round(best_score, 4), "detail": c_rec}
                contradicted_claims.append(entry)
                unsupported.append(entry)
            elif polarity_conflict_ev is not None:
                c_rec = {
                    "type": "polarity_contradiction",
                    "subject": cu.subject or cu.text[:40],
                    "claim_negated": cu.negated,
                    "evidence_negated": polarity_conflict_ev.negated,
                    "evidence_snippet": polarity_conflict_ev.text[:160],
                }
                contradictions.append(c_rec)
                entry = {"claim": cu.text, "reason": "negation_contradiction", "best_overlap": round(best_score, 4), "detail": c_rec}
                contradicted_claims.append(entry)
                unsupported.append(entry)
            elif best_score >= self.support_overlap_threshold and best_ev_unit is not None:
                supported.append(
                    {
                        "claim": cu.text,
                        "relation": cu.relation,
                        "overlap": round(best_score, 4),
                        "matched_evidence": best_ev_unit.text[:160],
                    }
                )
            else:
                entry = {"claim": cu.text, "reason": "missing_information", "best_overlap": round(best_score, 4)}
                missing_info_claims.append(entry)
                unsupported.append(entry)

        grounded_ratio = round(len(supported) / max(1, len(claim_clauses)), 4)
        if grounded_ratio >= 0.8 and not contradictions:
            risk = "LOW"
        elif grounded_ratio >= 0.4 and not contradicted_claims:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return {
            "claims_checked": len(claim_clauses),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "contradicted_claims": contradicted_claims,
            "missing_info_claims": missing_info_claims,
            "contradictions": contradictions,
            "clause_relations": [{"clause_id": c.clause_id, "relation": c.relation, "negated": c.negated} for c in claim_clauses],
            "grounded_ratio": grounded_ratio,
            "hallucination_risk": risk,
            "verified": len(unsupported) == 0 and len(contradictions) == 0,
        }
