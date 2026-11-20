"""JABRIX AI v3.4.0 — Hypothesis Generation, Evidence Comparison, and Unsupported Hypothesis Rejection."""
from __future__ import annotations

import re
from typing import Any

from nourai.memory.advanced_memory import _tokenize

_KV_RE = re.compile(r"([A-Za-z\u0600-\u06FF0-9 _-]{3,35})\s*(?:is|are|was|est|هو|هي|=)\s*([^\n.;,]{2,50})", re.I)


class HypothesisEngine:
    """Generates candidate hypotheses, evaluates them against evidence, and rejects unsupported ones."""

    def __init__(self, *, min_support_threshold: float = 0.25):
        self.min_support_threshold = float(min_support_threshold)

    def generate_and_evaluate(
        self,
        query: str,
        *,
        evidence: list[dict[str, Any]],
        causal_graph: dict[str, Any] | None = None,
        candidate_hypotheses: list[str] | None = None,
    ) -> dict[str, Any]:
        candidates: list[dict[str, str]] = []
        for h in (candidate_hypotheses or []):
            if h.strip():
                candidates.append({"statement": h.strip(), "origin": "user_or_caller"})

        # Generate hypotheses from causal chains
        for chain in (causal_graph or {}).get("causal_chains", [])[:3]:
            candidates.append(
                {
                    "statement": f"{chain[0]} leads to {chain[-1]}",
                    "origin": "causal_chain",
                }
            )

        # Generate evidence-grounded hypothesis from top retrieved evidence
        if evidence:
            top_txt = str(evidence[0].get("text") or evidence[0].get("content") or "").strip()
            if top_txt:
                candidates.append({"statement": top_txt[:180], "origin": "top_evidence"})
        elif not candidates:
            candidates.append(
                {
                    "statement": f"Direct unverified hypothesis for: {query.strip()[:120]}",
                    "origin": "query_fallback",
                }
            )

        evaluated: list[dict[str, Any]] = []
        supported_count = 0
        rejected_count = 0
        contested_count = 0

        for idx, cand in enumerate(candidates, start=1):
            stmt = cand["statement"]
            h_toks = set(_tokenize(stmt))
            h_kv = _KV_RE.search(stmt)
            h_subj = h_kv.group(1).strip().lower() if h_kv else None
            h_val = h_kv.group(2).strip().lower() if h_kv else None

            supporting: list[str] = []
            contradicting: list[str] = []
            max_overlap = 0.0

            for ev in evidence:
                ev_txt = str(ev.get("text") or ev.get("content") or "")
                ev_toks = set(_tokenize(ev_txt))
                overlap = len(h_toks & ev_toks) / max(1, len(h_toks)) if h_toks else 0.0
                if overlap > max_overlap:
                    max_overlap = overlap
                # Check direct factual contradiction on same subject with different value first
                ev_contradicts = False
                if h_subj and h_val:
                    for m in _KV_RE.finditer(ev_txt):
                        es = m.group(1).strip().lower()
                        ev_v = m.group(2).strip().lower()
                        if es == h_subj and ev_v != h_val:
                            contradicting.append(ev_txt[:140])
                            ev_contradicts = True

                if not ev_contradicts and overlap >= self.min_support_threshold:
                    supporting.append(ev_txt[:140])

            if contradicting and supporting:
                status = "CONTESTED"
                contested_count += 1
            elif contradicting:
                status = "REJECTED_CONTRADICTED"
                rejected_count += 1
            elif supporting and max_overlap >= self.min_support_threshold:
                status = "SUPPORTED"
                supported_count += 1
            else:
                status = "REJECTED_UNSUPPORTED"
                rejected_count += 1

            evaluated.append(
                {
                    "hypothesis_id": f"hyp_{idx}",
                    "statement": stmt,
                    "origin": cand["origin"],
                    "status": status,
                    "support_score": round(max_overlap, 4),
                    "supporting_count": len(supporting),
                    "contradicting_count": len(contradicting),
                    "supporting_evidence": supporting[:3],
                    "contradicting_evidence": contradicting[:3],
                }
            )

        return {
            "total_hypotheses": len(evaluated),
            "supported_count": supported_count,
            "rejected_count": rejected_count,
            "contested_count": contested_count,
            "hypotheses": evaluated,
        }
