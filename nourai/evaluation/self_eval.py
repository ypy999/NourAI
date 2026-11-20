"""Mission 5 — Self-Evaluation Engine: identifies failed tasks, classifies failure causes,
recommends improvements, generates candidate test cases, compares versions, and detects regressions.
Strictly prohibits automatic unverified model updates."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any


class SelfEvaluator:
    """Analyzes persisted tasks and evaluation runs to drive verifiable self-improvement."""

    allow_auto_unverified_model_updates: bool = False

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else None

    @staticmethod
    def classify_failure(error: str | None, trace: list[dict[str, Any]] | None = None) -> str:
        err = (error or "").lower()
        codes = {str(ev.get("error_code", "")).upper() for ev in (trace or []) if isinstance(ev, dict)}
        if codes & {"PERMISSION_DENIED", "PATH_VIOLATION", "SECRET_ACCESS_DENIED", "SQL_POLICY_VIOLATION", "SANDBOX_VIOLATION", "SSRF_BLOCKED"}:
            return "security_policy"
        if "APPROVAL_REQUIRED" in codes or "approval" in err:
            return "approval_gate"
        if "TIMEOUT" in codes or "timed out" in err:
            return "tool_timeout"
        if "NOT_FOUND" in codes or "not found" in err:
            return "missing_resource"
        if "plan" in err or "dependency" in err:
            return "planning_error"
        if "contradiction" in err or "verification" in err:
            return "verification_mismatch"
        return "runtime_or_unknown"

    @staticmethod
    def recommend_for_category(category: str) -> str:
        recommendations = {
            "security_policy": "Review permission grants or keep request blocked; never relax sandbox boundaries.",
            "approval_gate": "Route destructive tasks through the Control Center Approval Queue.",
            "tool_timeout": "Increase tool timeout or break large inputs into smaller chunks.",
            "missing_resource": "Verify workspace path existence or seed local memory/RAG fallback.",
            "planning_error": "Inspect PlanningEngine intent rules and DAG dependency ordering.",
            "verification_mismatch": "Audit Knowledge Graph contradictions and CriticEngine validation rules.",
            "runtime_or_unknown": "Inspect execution trace and add targeted unit tests for the failing input.",
        }
        return recommendations.get(category, recommendations["runtime_or_unknown"])

    def analyze_tasks(self, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze a list of task dicts (from TaskManager or benchmark) and generate diagnostics + candidate tests."""
        failed = [t for t in tasks if t.get("status") in ("FAILED", "CANCELLED")]
        category_counts: Counter[str] = Counter()
        candidate_tests: list[dict[str, Any]] = []
        failure_records: list[dict[str, Any]] = []

        for t in failed:
            cat = self.classify_failure(t.get("error"), t.get("trace"))
            category_counts[cat] += 1
            failure_records.append(
                {
                    "task_id": t.get("task_id"),
                    "objective": t.get("objective"),
                    "status": t.get("status"),
                    "failure_category": cat,
                    "error": t.get("error"),
                }
            )
            candidate_tests.append(
                {
                    "test_name": f"test_regression_{cat}_{str(t.get('task_id', 'x'))[-6:]}",
                    "objective": t.get("objective", ""),
                    "language": t.get("language", "english"),
                    "target_failure_category": cat,
                    "requires_human_review": True,
                }
            )

        recommendations = [
            {"category": cat, "count": cnt, "recommendation": self.recommend_for_category(cat)}
            for cat, cnt in category_counts.most_common()
        ]
        return {
            "total_tasks_analyzed": len(tasks),
            "failed_count": len(failed),
            "failure_rate": round(len(failed) / max(1, len(tasks)), 4),
            "failure_categories": dict(category_counts),
            "failures": failure_records,
            "recommendations": recommendations,
            "candidate_tests": candidate_tests,
            "allow_auto_unverified_model_updates": self.allow_auto_unverified_model_updates,
        }

    def analyze_from_db(self, *, limit: int = 200) -> dict[str, Any]:
        if not self.db_path or not self.db_path.is_file():
            return self.analyze_tasks([])
        with closing(sqlite3.connect(str(self.db_path), timeout=10.0)) as conn:
            rows = conn.execute(
                """
                SELECT task_id, user_id, objective, language, status, trace_json, error
                FROM v3_agent_tasks ORDER BY updated_at DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        tasks = [
            {
                "task_id": r[0],
                "user_id": r[1],
                "objective": r[2],
                "language": r[3],
                "status": r[4],
                "trace": json.loads(r[5]) if r[5] else [],
                "error": r[6],
            }
            for r in rows
        ]
        return self.analyze_tasks(tasks)

    @staticmethod
    def compare_versions(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two version evaluation metric dicts and flag any regressions."""
        higher_is_better = (
            "task_success_rate",
            "tool_selection_accuracy",
            "planning_quality",
            "memory_retrieval_accuracy",
            "failure_recovery_rate",
        )
        deltas: dict[str, float] = {}
        regressions: list[str] = []
        improvements: list[str] = []

        for metric in higher_is_better:
            if metric in baseline and metric in candidate:
                b_val = float(baseline[metric])
                c_val = float(candidate[metric])
                diff = round(c_val - b_val, 4)
                deltas[metric] = diff
                if diff < -1e-4:
                    regressions.append(f"{metric} regressed by {diff:+.4f} ({b_val} -> {c_val})")
                elif diff > 1e-4:
                    improvements.append(f"{metric} improved by {diff:+.4f} ({b_val} -> {c_val})")

        # Compare per-language perplexity (lower is better)
        b_nn = baseline.get("multilingual_neural_quality", {})
        c_nn = candidate.get("multilingual_neural_quality", {})
        for lang in ("arabic", "english", "french"):
            if lang in b_nn and lang in c_nn and "perplexity" in b_nn[lang] and "perplexity" in c_nn[lang]:
                bp = float(b_nn[lang]["perplexity"])
                cp = float(c_nn[lang]["perplexity"])
                diff = round(cp - bp, 4)
                deltas[f"{lang}_perplexity"] = diff
                if cp > bp * 1.05:
                    regressions.append(f"{lang} perplexity regressed ({bp} -> {cp})")
                elif cp < bp * 0.98:
                    improvements.append(f"{lang} perplexity improved ({bp} -> {cp})")

        return {
            "baseline_version": baseline.get("version", "unknown"),
            "candidate_version": candidate.get("version", "unknown"),
            "deltas": deltas,
            "improvements": improvements,
            "regressions": regressions,
            "regression_detected": len(regressions) > 0,
            "approved_for_promotion": len(regressions) == 0,
        }

    def compute_v32_monitoring_snapshot(
        self,
        *,
        research_reports: list[dict[str, Any]] | None = None,
        projects: list[dict[str, Any]] | None = None,
        memory_consistent: bool = True,
        eval_metrics: dict[str, Any] | None = None,
        hardware_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mission 6: Track all 9 v3.2 self-monitoring KPIs and generate actionable diagnostics."""
        r_list = list(research_reports or [])
        p_list = list(projects or [])
        em = dict(eval_metrics or {})

        # Research & citation metrics
        research_with_facts = sum(1 for r in r_list if r.get("verified_facts"))
        research_acc = round(research_with_facts / len(r_list), 4) if r_list else 1.0
        citation_ok = sum(1 for r in r_list if float(r.get("citation_coverage", 0.0)) >= 0.99 or not r.get("verified_facts"))
        citation_corr = round(citation_ok / len(r_list), 4) if r_list else 1.0

        # Project completion rate
        if p_list:
            proj_comp = round(sum(float(p.get("progress", {}).get("completion_pct", 0.0)) for p in p_list) / (len(p_list) * 100.0), 4)
        else:
            proj_comp = 1.0

        diagnostics: list[str] = []
        if not memory_consistent:
            diagnostics.append("Memory contradictions detected; review Semantic Memory in Control Center.")
        if research_acc < 0.85:
            diagnostics.append("Research evidence coverage below 85%; ingest primary documents for missing sub-questions.")
        if proj_comp < 0.5 and p_list:
            diagnostics.append("Active projects below 50% completion; inspect blocked or approval-pending tasks.")

        return {
            "version": "3.2.0",
            "allow_auto_unverified_model_updates": self.allow_auto_unverified_model_updates,
            "metrics": {
                "research_accuracy": research_acc,
                "citation_correctness": citation_corr,
                "project_completion_rate": proj_comp,
                "planning_accuracy": float(em.get("planning_quality", 1.0)),
                "tool_selection": float(em.get("tool_selection_accuracy", 1.0)),
                "memory_consistency": bool(memory_consistent),
                "recovery_performance": float(em.get("failure_recovery_rate", 1.0)),
                "neural_generation_quality": em.get("multilingual_neural_quality", {}),
                "cpu_ram_consumption": hardware_snapshot or {},
            },
            "actionable_diagnostics": diagnostics or ["All v3.2 self-monitoring KPIs are healthy."],
        }
