"""Mission 2 — Realistic 105-Task Real-World Autonomous Agent Benchmark with Independent Verification.

Covers 11 categories (>= 100 tasks):
1. multi_step (10)
2. arabic_document_analysis (10)
3. english_research (10)
4. french_document_processing (10)
5. file_organization (10)
6. report_creation (10)
7. knowledge_retrieval (10)
8. multi_tool_workflows (10)
9. error_recovery (10)
10. interrupted_and_approval_recovery (8)
11. long_running_and_parallel_dag (7)
Total = 105 independently verified tasks.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any

from nourai.agent.orchestrator import AutonomousAgentOrchestrator
from nourai.agent.planning import ExecutionPlan, PlanStep
from nourai.agent.task_manager import TaskManager
from nourai.knowledge.knowledge_graph import KnowledgeGraph
from nourai.memory.advanced_memory import AdvancedMemoryManager
from nourai.tools.builtin_tools import register_default_tools
from nourai.tools.tool_engine import PermissionPolicy, ToolRegistry


def _seed_workspace(root: Path, mm: AdvancedMemoryManager, kg: KnowledgeGraph, uid: str) -> None:
    """Seed realistic multilingual workspace files, memories, and Knowledge Graph facts."""
    docs = root / "docs_sample"
    docs.mkdir(parents=True, exist_ok=True)
    for i in range(1, 11):
        (docs / f"ar_doc_{i}.txt").write_text(
            f"تقرير رقم {i}: يعمل نظام نور للذكاء الاصطناعي محلياً لحماية الخصوصية وتحليل البيانات العربية بدقة عالية.",
            encoding="utf-8",
        )
        (docs / f"fr_doc_{i}.txt").write_text(
            f"Rapport numéro {i} : NourAI traite les documents en français localement avec mémoire structurée et sécurité.",
            encoding="utf-8",
        )
        (docs / f"en_file_{i}.md").write_text(
            f"# Research Note {i}\nNourAI v3.1 supports local-first autonomous planning, DAG execution, and SQLite WAL persistence.\n",
            encoding="utf-8",
        )

    facts = [
        ("NourAI uses PyTorch and SQLite WAL for local persistence.", "en"),
        ("The capital of Qatar is Doha.", "en"),
        ("The capital of France is Paris.", "en"),
        ("عاصمة قطر هي الدوحة.", "ar"),
        ("عاصمة مصر هي القاهرة.", "ar"),
        ("La capitale du Maroc est Rabat.", "fr"),
        ("NourAI v3.1 introduces DAG parallel read-only tool orchestration.", "en"),
        ("L2AnchorRegularizer prevents catastrophic forgetting during SFT.", "en"),
        ("ByteLevelBPETokenizer compresses Arabic text losslessly.", "en"),
        ("Approval checkpoints protect destructive file write operations.", "en"),
    ]
    for fact, _ in facts:
        mm.store(uid, "semantic", fact, source="benchmark_seed")
        kg.ingest_text(fact, source="benchmark_seed")


def run_105_task_benchmark(workspace_root: str | Path) -> dict[str, Any]:
    """Execute all 105 real-world tasks and independently verify every expected outcome."""
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "bench_105.sqlite3"
    mm = AdvancedMemoryManager(db_path)
    kg = KnowledgeGraph(db_path)
    policy = PermissionPolicy()
    tools = ToolRegistry(db_path=db_path, policy=policy)
    register_default_tools(tools, workspace_root=root, db_path=db_path, memory_manager=mm, knowledge_graph=kg)
    tm = TaskManager(db_path)
    orch = AutonomousAgentOrchestrator(
        memory_manager=mm,
        knowledge_graph=kg,
        tool_registry=tools,
        task_manager=tm,
    )
    uid = "bench_105_user"
    policy.grant(uid, "fs:write")
    _seed_workspace(root, mm, kg, uid)

    results: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    started_all = time.perf_counter()

    def record(tid: str, category: str, passed: bool, expected: str, observed: str, latency_ms: float) -> None:
        by_category[category]["total"] += 1
        if passed:
            by_category[category]["passed"] += 1
        results.append(
            {
                "task_id": tid,
                "category": category,
                "passed": passed,
                "expected": expected,
                "observed": observed[:160],
                "latency_ms": round(latency_ms, 2),
            }
        )

    # 1. Multi-step math & structured computation (10 tasks)
    for i in range(1, 11):
        expr = f"({i * 10} + {i * 5}) * {i + 1}"
        expected_val = (i * 10 + i * 5) * (i + 1)
        t0 = time.perf_counter()
        task = orch.run_task(uid, f"Calculate {expr} and verify the result")
        resp = (task.get("result") or {}).get("response", "")
        ok = task["status"] == "COMPLETED" and str(expected_val) in resp
        record(f"ms_{i:02d}", "multi_step", ok, str(expected_val), resp, (time.perf_counter() - t0) * 1000)

    # 2. Arabic document analysis (10 tasks)
    for i in range(1, 11):
        t0 = time.perf_counter()
        task = orch.run_task(
            uid,
            f"حلل المستند docs_sample/ar_doc_{i}.txt واستخرج عدد الكلمات",
            language_hint="arabic",
        )
        outs = (task.get("result") or {}).get("step_outputs", [])
        doc_out = next((o["output"] for o in outs if o["tool_name"] == "document_analysis"), {})
        ok = task["status"] == "COMPLETED" and doc_out.get("language") == "arabic" and int(doc_out.get("word_count", 0)) >= 10
        record(f"ar_doc_{i:02d}", "arabic_document_analysis", ok, "language=arabic, words>=10", str(doc_out), (time.perf_counter() - t0) * 1000)

    # 3. English research tasks (10 tasks)
    en_queries = [
        ("What do you know about PyTorch and SQLite WAL?", "SQLite"),
        ("What do you know about the capital of Qatar?", "Doha"),
        ("What do you know about the capital of France?", "Paris"),
        ("What do you know about DAG parallel tool orchestration?", "DAG"),
        ("What do you know about L2AnchorRegularizer?", "forgetting"),
        ("What do you know about ByteLevelBPETokenizer?", "Arabic"),
        ("What do you know about approval checkpoints?", "Approval"),
        ("Read file docs_sample/en_file_1.md", "Research Note 1"),
        ("Read file docs_sample/en_file_2.md", "Research Note 2"),
        ("SELECT count(*) FROM v3_memory_items", "SQL query returned 1 row"),
    ]
    for i, (q, needle) in enumerate(en_queries, start=1):
        t0 = time.perf_counter()
        task = orch.run_task(uid, q, language_hint="english")
        resp = (task.get("result") or {}).get("response", "")
        ok = task["status"] == "COMPLETED" and needle.lower() in resp.lower()
        record(f"en_res_{i:02d}", "english_research", ok, f"contains '{needle}'", resp, (time.perf_counter() - t0) * 1000)

    # 4. French document processing (10 tasks)
    for i in range(1, 11):
        t0 = time.perf_counter()
        task = orch.run_task(
            uid,
            f"Analyser le document docs_sample/fr_doc_{i}.txt",
            language_hint="french",
        )
        outs = (task.get("result") or {}).get("step_outputs", [])
        doc_out = next((o["output"] for o in outs if o["tool_name"] == "document_analysis"), {})
        ok = task["status"] == "COMPLETED" and doc_out.get("language") == "french" and int(doc_out.get("word_count", 0)) >= 10
        record(f"fr_doc_{i:02d}", "french_document_processing", ok, "language=french, words>=10", str(doc_out), (time.perf_counter() - t0) * 1000)

    # 5. File organization & approved writes with disk verification (10 tasks)
    for i in range(1, 11):
        rel_path = f"organized/item_{i:02d}.txt"
        payload_text = f"Organized file #{i} verified by NourAI v3.1"
        t0 = time.perf_counter()
        task = orch.run_task(
            uid,
            f"Write file {rel_path}",
            explicit_params={"path": rel_path, "content": payload_text},
            approved=True,
        )
        disk_ok = (root / rel_path).is_file() and (root / rel_path).read_text(encoding="utf-8") == payload_text
        ok = task["status"] == "COMPLETED" and disk_ok
        record(f"file_org_{i:02d}", "file_organization", ok, f"file '{rel_path}' exact content", str(disk_ok), (time.perf_counter() - t0) * 1000)

    # 6. Report creation (10 tasks)
    for i in range(1, 11):
        t0 = time.perf_counter()
        task = orch.run_task(uid, f"Generate report on NourAI subsystem module {i}")
        resp = (task.get("result") or {}).get("response", "")
        ok = task["status"] == "COMPLETED" and "# NourAI Report" in resp and "## Objective" in resp
        record(f"rep_{i:02d}", "report_creation", ok, "Markdown report with headings", resp, (time.perf_counter() - t0) * 1000)

    # 7. Knowledge retrieval (10 tasks across AR/EN/FR)
    kr_cases = [
        ("تذكر عاصمة قطر", "الدوحة", "arabic"),
        ("تذكر عاصمة مصر", "القاهرة", "arabic"),
        ("Mémoire capitale du Maroc", "Rabat", "french"),
        ("Remember SQLite WAL persistence", "SQLite", "english"),
        ("Remember PyTorch", "PyTorch", "english"),
        ("Remember L2AnchorRegularizer", "L2AnchorRegularizer", "english"),
        ("Remember ByteLevelBPETokenizer", "ByteLevelBPETokenizer", "english"),
        ("Remember DAG parallel", "DAG", "english"),
        ("Remember Approval checkpoints", "Approval", "english"),
        ("Remember capital of France", "Paris", "english"),
    ]
    for i, (q, expected_word, lang) in enumerate(kr_cases, start=1):
        t0 = time.perf_counter()
        task = orch.run_task(uid, q, language_hint=lang)
        resp = (task.get("result") or {}).get("response", "")
        ok = task["status"] == "COMPLETED" and expected_word.lower() in resp.lower()
        record(f"kr_{i:02d}", "knowledge_retrieval", ok, f"contains '{expected_word}'", resp, (time.perf_counter() - t0) * 1000)

    # 8. Multi-tool workflows: file_read + document_analysis chained (10 tasks)
    for i in range(1, 11):
        t0 = time.perf_counter()
        task = orch.run_task(uid, f"Inspect file docs_sample/en_file_{i}.md")
        tools_used = (task.get("result") or {}).get("tools_used", [])
        ok = task["status"] == "COMPLETED" and "file_read" in tools_used and "document_analysis" in tools_used
        record(f"mtw_{i:02d}", "multi_tool_workflows", ok, "file_read + document_analysis", str(tools_used), (time.perf_counter() - t0) * 1000)

    # 9. Error recovery: missing file -> automatic fallback to knowledge_retrieval (10 tasks)
    for i in range(1, 11):
        t0 = time.perf_counter()
        task = orch.run_task(uid, f"Read file missing_doc_{i}.txt")
        res = task.get("result") or {}
        ok = task["status"] == "COMPLETED" and len(res.get("recoveries", [])) >= 1 and "knowledge_retrieval" in res.get("tools_used", [])
        record(f"rec_{i:02d}", "error_recovery", ok, "recovered via knowledge_retrieval", str(res.get("tools_used")), (time.perf_counter() - t0) * 1000)

    # 10. Interrupted (cancelled) & human-approval checkpoint recovery (8 tasks)
    for i in range(1, 5):
        # 4 approval-gated tasks paused -> resumed
        t0 = time.perf_counter()
        rel_p = f"approved/doc_{i}.txt"
        t_pause = orch.run_task(uid, f"Write file {rel_p}", explicit_params={"path": rel_p, "content": f"ok_{i}"}, approved=False)
        paused_ok = t_pause["status"] == "AWAITING_APPROVAL"
        t_resumed = orch.approve_task(t_pause["task_id"], uid) if paused_ok else t_pause
        ok = paused_ok and t_resumed["status"] == "COMPLETED" and (root / rel_p).read_text(encoding="utf-8") == f"ok_{i}"
        record(f"appr_{i:02d}", "interrupted_and_approval_recovery", ok, "AWAITING_APPROVAL -> COMPLETED", t_resumed["status"], (time.perf_counter() - t0) * 1000)

    for i in range(5, 9):
        # 4 cancelled tasks verified as cleanly CANCELLED in SQLite
        t0 = time.perf_counter()
        t_new = tm.create_task(uid, f"Interrupted long job {i}")
        cancelled = tm.request_cancel(t_new["task_id"], user_id=uid)
        reloaded = tm.get_task(t_new["task_id"], user_id=uid)
        ok = cancelled and reloaded is not None and reloaded["status"] == "CANCELLED"
        record(f"canc_{i:02d}", "interrupted_and_approval_recovery", ok, "status=CANCELLED", str(reloaded["status"] if reloaded else None), (time.perf_counter() - t0) * 1000)

    # 11. Long-running / parallel DAG multi-wave execution (7 tasks)
    for i in range(1, 8):
        t0 = time.perf_counter()
        plan = ExecutionPlan(
            objective=f"Parallel DAG workflow #{i}",
            strategy="parallel_dag_benchmark",
            max_steps=6,
            steps=[
                PlanStep(step_id="s1", description="Parallel calc A", tool_name="python_sandbox", tool_input={"expression": f"{i} * 100"}),
                PlanStep(step_id="s2", description="Parallel read B", tool_name="file_read", tool_input={"path": f"docs_sample/en_file_{i}.md"}),
                PlanStep(step_id="s3", description="Parallel KB C", tool_name="knowledge_retrieval", tool_input={"query": "SQLite WAL", "top_k": 2}),
                PlanStep(
                    step_id="s4",
                    description="Dependent final report",
                    tool_name="report_generation",
                    tool_input={"title": f"DAG Report {i}", "sections": [{"heading": "Summary", "body": "All parallel branches completed."}]},
                    depends_on=["s1", "s2", "s3"],
                ),
            ],
        )
        dag_res = orch.dag_executor.execute_plan_dag(plan, user_id=uid, approved=False)
        ok = (
            dag_res["parallel_waves"] >= 1
            and len(dag_res["step_results"]) == 4
            and all(r.ok for r in dag_res["step_results"])
        )
        record(f"dag_{i:02d}", "long_running_and_parallel_dag", ok, "parallel_waves>=1 & 4/4 steps ok", f"waves={dag_res['parallel_waves']}, ok={sum(r.ok for r in dag_res['step_results'])}/4", (time.perf_counter() - t0) * 1000)

    total_tasks = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    total_ms = round((time.perf_counter() - started_all) * 1000.0, 2)
    avg_ms = round(sum(r["latency_ms"] for r in results) / max(1, total_tasks), 2)

    return {
        "benchmark_version": "3.1.0",
        "total_tasks": total_tasks,
        "total_passed": total_passed,
        "total_failed": total_tasks - total_passed,
        "success_rate": round(total_passed / max(1, total_tasks), 4),
        "total_elapsed_ms": total_ms,
        "avg_task_latency_ms": avg_ms,
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "failed_tasks": [r for r in results if not r["passed"]],
    }
