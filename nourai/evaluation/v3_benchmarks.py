"""Measurable v3.0 Evaluation Suite, Fixed Datasets, Regression Gate, and SQLite Report Persistence."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any

import torch

from model.transformer import GPT
from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.monitoring.hardware import HardwareMonitor
from tokenizer.byte_tokenizer import ByteTokenizer

FIXED_MULTILINGUAL_SAMPLES: dict[str, list[str]] = {
    "arabic": [
        "نور للذكاء الاصطناعي نموذج لغوي محلي متعدد اللغات يدعم العربية والإنجليزية والفرنسية.",
        "تحليل المستندات المحلية والذاكرة الدائمة يتيحان استرجاع المعرفة بأمان دون خوادم خارجية.",
        "التحقق من النتائج والتعافي من الأخطاء ضروريان لبناء وكيل ذكي موثوق.",
    ],
    "english": [
        "NourAI is a local-first multilingual language model and autonomous agent supporting Arabic, English, and French.",
        "Structured memory tiers and a bidirectional knowledge graph enable verifiable evidence-based reasoning.",
        "Least-privilege tool permissions and sandboxing prevent unauthorized or destructive actions.",
    ],
    "french": [
        "NourAI est un modèle de langage local et un agent autonome multilingue pour l'arabe, l'anglais et le français.",
        "La mémoire structurée et le graphe de connaissances assurent un raisonnement fondé sur des preuves.",
        "Les autorisations minimales et le bac à sable garantissent une exécution sécurisée des outils.",
    ],
}

FIXED_AGENT_BENCHMARK: tuple[dict[str, Any], ...] = (
    {
        "id": "bench_calc_en",
        "objective": "Calculate (125 * 4) + 50",
        "language": "english",
        "expected_tool": "python_sandbox",
        "expected_substring": "550",
    },
    {
        "id": "bench_calc_ar",
        "objective": "احسب ناتج 45 * 12",
        "language": "arabic",
        "expected_tool": "python_sandbox",
        "expected_substring": "540",
    },
    {
        "id": "bench_doc_fr",
        "objective": "Analyser le texte suivant: NourAI fonctionne localement en arabe, anglais et français.",
        "language": "french",
        "expected_tool": "document_analysis",
        "expected_substring": "mots",
    },
    {
        "id": "bench_report_en",
        "objective": "Generate report on NourAI v3 architecture status",
        "language": "english",
        "expected_tool": "report_generation",
        "expected_substring": "# NourAI Report",
    },
    {
        "id": "bench_recovery_missing_file",
        "objective": "Read file nonexistent_demo_file_xyz.txt",
        "language": "english",
        "expected_tool": "knowledge_retrieval",  # expected via RecoveryEngine fallback
        "expect_recovery": True,
    },
)


def _evaluate_neural_language_loss(model: GPT, tokenizer: ByteTokenizer, texts: list[str]) -> dict[str, float]:
    model.eval()
    ctx = model.config.context_length
    losses: list[float] = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                continue
            ids = ids[:ctx]
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor([ids[1:]], dtype=torch.long)
            _, loss = model(x, y)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    avg_loss = sum(losses) / max(1, len(losses))
    import math
    ppl = math.exp(min(20.0, avg_loss))
    bpb = avg_loss / math.log(2.0)
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 4),
        "bits_per_byte": round(bpb, 4),
    }


def check_regression_gate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    *,
    min_task_success_rate: float = 0.80,
    min_tool_accuracy: float = 0.80,
    min_memory_accuracy: float = 0.80,
) -> dict[str, Any]:
    """Verify candidate metrics meet absolute thresholds and do not regress vs baseline."""
    failures: list[str] = []
    if candidate_metrics.get("task_success_rate", 0.0) < min_task_success_rate:
        failures.append(f"task_success_rate {candidate_metrics.get('task_success_rate')} < {min_task_success_rate}")
    if candidate_metrics.get("tool_selection_accuracy", 0.0) < min_tool_accuracy:
        failures.append(f"tool_selection_accuracy {candidate_metrics.get('tool_selection_accuracy')} < {min_tool_accuracy}")
    if candidate_metrics.get("memory_retrieval_accuracy", 0.0) < min_memory_accuracy:
        failures.append(f"memory_retrieval_accuracy {candidate_metrics.get('memory_retrieval_accuracy')} < {min_memory_accuracy}")

    if baseline_metrics:
        if candidate_metrics.get("task_success_rate", 0.0) + 1e-6 < baseline_metrics.get("task_success_rate", 0.0):
            failures.append("task_success_rate regressed below baseline")
        if candidate_metrics.get("memory_retrieval_accuracy", 0.0) + 1e-6 < baseline_metrics.get("memory_retrieval_accuracy", 0.0):
            failures.append("memory_retrieval_accuracy regressed below baseline")

    return {"passed": len(failures) == 0, "failures": failures}


class V3EvaluationSuite:
    """Runs measurable benchmarks across Agent, Memory, Recovery, Hardware, and Neural Model."""

    allow_auto_train_on_unverified_feedback: bool = False

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        if not self.db_path:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    passed_regression INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def run_full_evaluation(
        self,
        orchestrator: AutonomousAgentOrchestrator,
        *,
        model: GPT | None = None,
        version: str = "3.0.0",
        baseline_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bench_user = "eval_benchmark_user"
        # Seed deterministic semantic memory for memory retrieval accuracy test
        orchestrator.memory.clear_user_memories(bench_user)
        orchestrator.memory.store(bench_user, "semantic", "NourAI v3 uses four memory tiers and SQLite WAL persistence.", source="benchmark")
        orchestrator.memory.store(bench_user, "semantic", "عاصمة قطر هي الدوحة.", source="benchmark")
        orchestrator.memory.store(bench_user, "semantic", "La capitale de la France est Paris.", source="benchmark")

        mem_queries = [
            ("SQLite WAL persistence", "SQLite"),
            ("عاصمة قطر", "الدوحة"),
            ("capitale de la France", "Paris"),
        ]
        mem_hits = 0
        for q, expected in mem_queries:
            res = orchestrator.memory.semantic_search(bench_user, q, top_k=2)
            if any(expected.lower() in r["content"].lower() for r in res):
                mem_hits += 1
        memory_retrieval_accuracy = round(mem_hits / len(mem_queries), 4)

        # Run fixed agent benchmark tasks
        success_count = 0
        tool_matches = 0
        plan_valid_count = 0
        recovery_expected = 0
        recovery_succeeded = 0
        latencies: list[float] = []
        task_details: list[dict[str, Any]] = []

        for case in FIXED_AGENT_BENCHMARK:
            task = orchestrator.run_task(bench_user, case["objective"], language_hint=case["language"])
            res = task.get("result") or {}
            ok = task["status"] == "COMPLETED"
            if ok:
                success_count += 1
            tools_used = res.get("tools_used", [])
            if case["expected_tool"] in tools_used:
                tool_matches += 1
            if task.get("plan", {}).get("steps"):
                plan_valid_count += 1
            if case.get("expect_recovery"):
                recovery_expected += 1
                if ok and res.get("recoveries"):
                    recovery_succeeded += 1
            if "latency_ms" in res:
                latencies.append(float(res["latency_ms"]))
            task_details.append(
                {
                    "id": case["id"],
                    "status": task["status"],
                    "tools_used": tools_used,
                    "recoveries": len(res.get("recoveries", [])),
                    "latency_ms": res.get("latency_ms"),
                }
            )

        # Neural multilingual metrics (honest measurement on provided model)
        neural_metrics: dict[str, Any] = {}
        if model is not None:
            tok = ByteTokenizer()
            for lang, texts in FIXED_MULTILINGUAL_SAMPLES.items():
                neural_metrics[lang] = _evaluate_neural_language_loss(model, tok, texts)

        hw = HardwareMonitor().snapshot().to_dict()
        total_cases = len(FIXED_AGENT_BENCHMARK)
        metrics: dict[str, Any] = {
            "version": version,
            "task_success_rate": round(success_count / max(1, total_cases), 4),
            "tool_selection_accuracy": round(tool_matches / max(1, total_cases), 4),
            "planning_quality": round(plan_valid_count / max(1, total_cases), 4),
            "memory_retrieval_accuracy": memory_retrieval_accuracy,
            "failure_recovery_rate": round(recovery_succeeded / max(1, recovery_expected), 4) if recovery_expected else 1.0,
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
            "hardware": hw,
            "multilingual_neural_quality": neural_metrics,
            "auto_train_on_unverified_feedback": self.allow_auto_train_on_unverified_feedback,
            "task_details": task_details,
        }

        gate = check_regression_gate(metrics, baseline_metrics)
        run_id = f"eval_{secrets.token_hex(6)}"
        report = {
            "run_id": run_id,
            "version": version,
            "metrics": metrics,
            "regression_gate": gate,
            "created_at": time.time(),
        }
        if self.db_path:
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    """
                    INSERT INTO v3_evaluation_runs(run_id, version, metrics_json, passed_regression, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, version, json.dumps(metrics, ensure_ascii=False), 1 if gate["passed"] else 0, report["created_at"]),
                )
        return report

    def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.db_path or not self.db_path.is_file():
            return []
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT run_id, version, metrics_json, passed_regression, created_at FROM v3_evaluation_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "version": r[1],
                "metrics": json.loads(r[2]),
                "passed_regression": bool(r[3]),
                "created_at": float(r[4]),
            }
            for r in rows
        ]
