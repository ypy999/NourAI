"""JABRIX AI v3.4.0 — Confidence Calibration, Uncertainty Estimation, and Missing Evidence Detection."""
from __future__ import annotations

from typing import Any


class UncertaintyEngine:
    """Separates epistemic uncertainty (missing evidence) from conflict uncertainty (contradictions)."""

    def estimate(
        self,
        *,
        sub_goals: list[str],
        evidence: list[dict[str, Any]],
        hypothesis_report: dict[str, Any],
        verification_report: dict[str, Any],
    ) -> dict[str, Any]:
        missing_evidence: list[str] = []
        if not evidence:
            missing_evidence.append("No relevant local memory, document, or knowledge-graph evidence retrieved.")
        elif len(evidence) < len(sub_goals):
            missing_evidence.append(
                f"Retrieved {len(evidence)} evidence item(s) for {len(sub_goals)} sub-goal(s); partial coverage only."
            )

        unsupported_claims = verification_report.get("unsupported_claims", [])
        if unsupported_claims:
            missing_evidence.append(f"{len(unsupported_claims)} claim(s) lack direct supporting evidence.")

        # Epistemic uncertainty (lack of evidence)
        grounded_ratio = float(verification_report.get("grounded_ratio", 0.0))
        ev_factor = min(1.0, len(evidence) / max(1, len(sub_goals)))
        epistemic_uncertainty = round(max(0.0, min(1.0, 1.0 - (0.6 * grounded_ratio + 0.4 * ev_factor))), 4)

        # Conflict uncertainty (contradictions)
        contradictions = len(verification_report.get("contradictions", [])) + int(hypothesis_report.get("contested_count", 0))
        conflict_uncertainty = round(min(1.0, 0.3 * contradictions), 4)

        total_uncertainty = round(min(1.0, max(epistemic_uncertainty, 0.65 * epistemic_uncertainty + 0.55 * conflict_uncertainty)), 4)
        calibrated_confidence = round(max(0.05, min(0.98, 1.0 - total_uncertainty)), 4)

        if calibrated_confidence >= 0.75:
            band = "HIGH"
        elif calibrated_confidence >= 0.45:
            band = "MEDIUM"
        else:
            band = "LOW"

        return {
            "calibrated_confidence": calibrated_confidence,
            "confidence_band": band,
            "epistemic_uncertainty": epistemic_uncertainty,
            "conflict_uncertainty": conflict_uncertainty,
            "total_uncertainty": total_uncertainty,
            "missing_evidence": missing_evidence,
        }
