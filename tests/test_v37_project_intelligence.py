"""JABRIX AI v3.7.0 — Comprehensive Deep Project Intelligence & Autonomous Optimization Tests.

Covers all 10 new engines, incremental indexing vs full indexing, cross-file
analysis on multi-module dependencies, project memory versioning/invalidation,
root-cause ranking, optimization planning, release readiness, lazy loading,
8-agent MasterController integration, and v3.7 REST API endpoints.
"""
from __future__ import annotations

from pathlib import Path
import time

from nourai.api.v3_routes import V3APIRouter
from nourai.project_intel import (
    ArchitectureIntelligenceEngine,
    AutoTestIntelligenceEngine,
    ChangeImpactEngine,
    CrossFileReasoningEngine,
    IncrementalProjectIndex,
    OptimizationPlannerEngine,
    PerformanceProfilerEngine,
    ProjectIntelligenceHub,
    ProjectMemoryEngine,
    ProjectQualityEngine,
    ReleaseIntelligenceEngine,
    RootCauseEngine,
)


def _create_multi_layer_project(root: Path) -> Path:
    proj = root / "deep_proj"
    for d in ("app/core", "app/service", "app/api", "tests"):
        (proj / d).mkdir(parents=True)
        if d.startswith("app"):
            (proj / d / "__init__.py").write_text("", encoding="utf-8")
    (proj / "app" / "__init__.py").write_text("", encoding="utf-8")

    (proj / "app" / "core" / "math_core.py").write_text(
        '"""Core math utilities."""\n'
        "def compute_score(items, limit=10):\n"
        '    """Compute nested score."""\n'
        "    out = ''\n"
        "    for a in items:\n"
        "        for b in items:\n"
        "            out += str(a + b)\n"
        "    return out[:limit]\n",
        encoding="utf-8",
    )
    (proj / "app" / "service" / "processor.py").write_text(
        "from app.core.math_core import compute_score\n\n"
        "def process_batch(items):\n"
        "    return compute_score(items, limit=20)\n",
        encoding="utf-8",
    )
    (proj / "app" / "api" / "routes.py").write_text(
        "from app.service.processor import process_batch\n\n"
        "def handle_request(payload):\n"
        "    route = '/api/v3/demo/process'\n"
        "    return process_batch(payload.get('items', []))\n",
        encoding="utf-8",
    )
    (proj / "tests" / "test_processor.py").write_text(
        "from app.service.processor import process_batch\n\n"
        "def test_process_batch():\n"
        "    assert process_batch([1]) == '2'\n",
        encoding="utf-8",
    )
    return proj


def test_incremental_indexing_reuses_cache_and_bounds_memory(tmp_path: Path):
    proj = _create_multi_layer_project(tmp_path)
    idx = IncrementalProjectIndex(max_cached_files=50)

    r1 = idx.update(proj)
    assert r1["files_total"] >= 4
    assert r1["files_reparsed"] == r1["files_total"]
    assert r1["files_reused_from_cache"] == 0

    # No changes -> 0 reparsed, 100% reused from cache
    r2 = idx.update(proj)
    assert r2["files_reparsed"] == 0
    assert r2["files_reused_from_cache"] == r1["files_total"]

    # Modify exactly 1 file -> only 1 file reparsed
    time.sleep(0.02)
    target = proj / "app" / "service" / "processor.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# minor comment\n", encoding="utf-8")
    r3 = idx.update(proj)
    assert r3["files_reparsed"] == 1
    assert r3["files_reused_from_cache"] == r1["files_total"] - 1


def test_architecture_and_cross_file_reasoning(tmp_path: Path):
    proj = _create_multi_layer_project(tmp_path)
    # Inject a deliberate layer boundary violation: core importing api
    (proj / "app" / "core" / "bad_import.py").write_text("import app.api.routes\n", encoding="utf-8")

    idx = IncrementalProjectIndex()
    arch = ArchitectureIntelligenceEngine(index=idx).analyze(proj)
    assert arch["components_count"] >= 3
    assert len(arch["layer_boundary_violations"]) >= 1

    cf = CrossFileReasoningEngine(index=idx).analyze(proj)
    assert cf["cross_file_call_edges_count"] >= 2
    callees = {e["callee"] for e in cf["cross_file_call_edges"]}
    assert "compute_score" in callees
    assert "process_batch" in callees
    assert any("/api/v3/demo/process" in a["routes"] for a in cf["api_dependencies"])


def test_project_memory_versioning_and_invalidation(tmp_path: Path):
    pm = ProjectMemoryEngine(tmp_path / "pm.db")
    v1 = pm.record(
        project_key="proj_1",
        record_type="ADR",
        title="Use SQLite WAL",
        content="Use SQLite WAL mode for local persistence.",
        related_files=["app/core/math_core.py"],
    )
    assert v1["version"] == 1 and v1["active"] is True

    v2 = pm.record(
        project_key="proj_1",
        record_type="ADR",
        title="Use SQLite WAL",
        content="Use SQLite WAL mode with 10s busy timeout.",
        related_files=["app/core/math_core.py"],
    )
    assert v2["version"] == 2
    active = pm.list_records("proj_1", active_only=True)
    assert len(active) == 1 and active[0]["version"] == 2

    inv = pm.invalidate_for_changed_files("proj_1", ["app/core/math_core.py"])
    assert inv["invalidated_count"] == 1
    assert len(pm.list_records("proj_1", active_only=True)) == 0


def test_change_impact_root_cause_and_auto_test_intelligence(tmp_path: Path):
    proj = _create_multi_layer_project(tmp_path)
    idx = IncrementalProjectIndex()

    impact_eng = ChangeImpactEngine(index=idx)
    syms = impact_eng.identify_changed_symbols(
        "def a(): return 1\ndef b(): return 2\n",
        "def a(): return 1\ndef b(): return 99\n",
    )
    assert syms == ["b"]

    impact = impact_eng.analyze_impact(proj, changed_files=["app/core/math_core.py"])
    assert "app/service/processor.py" in impact["downstream_impacted_files"]
    assert "app/api/routes.py" in impact["downstream_impacted_files"]
    assert "tests/test_processor.py" in impact["impacted_tests"]

    rc = RootCauseEngine(index=idx).diagnose(
        proj,
        error_text='Traceback:\n  File "app/service/processor.py", line 4, in process_batch\nZeroDivisionError',
        changed_files=["app/core/math_core.py"],
    )
    assert rc["top_root_cause"] is not None
    ranked_files = [r["file"] for r in rc["ranked_root_causes"]]
    assert "app/service/processor.py" in ranked_files
    assert "app/core/math_core.py" in ranked_files

    ati = AutoTestIntelligenceEngine(index=idx, impact_engine=impact_eng)
    prio = ati.prioritize_regression_tests(proj, changed_files=["app/core/math_core.py"])
    assert prio["high_priority_count"] >= 1
    assert prio["prioritized_tests"][0]["test_file"] == "tests/test_processor.py"

    gaps = ati.analyze_gaps_and_edge_cases(proj)
    assert len(gaps["edge_case_recommendations"]) >= 1
    ft = ati.generate_failure_driven_test(rc)
    assert ft["generated"] is True and "def test_regression_" in ft["test_code"]


def test_profiler_optimization_planner_quality_and_release_intelligence(tmp_path: Path):
    proj = _create_multi_layer_project(tmp_path)
    idx = IncrementalProjectIndex()

    prof = PerformanceProfilerEngine(index=idx)
    sp = prof.static_profile(proj)
    assert sp["cpu_bottlenecks_count"] >= 1
    assert sp["memory_bottlenecks_count"] >= 1
    assert sp["executed_untrusted_code"] is False

    # Before/after micro-benchmark comparison
    b_res = prof.profile_callable(lambda: sum(range(2000)), iterations=5)
    a_res = prof.profile_callable(lambda: 1999 * 2000 // 2, iterations=5)
    cmp = prof.compare_before_after(b_res, a_res)
    assert cmp["improved"] is True and cmp["speedup_x"] > 1.0

    plan = OptimizationPlannerEngine(index=idx, profiler=prof).create_plan(proj)
    assert plan["recommendations_count"] >= 1
    assert len(plan["resource_aware_plan"]["steps"]) == 5

    qual = ProjectQualityEngine(index=idx).evaluate(proj)
    assert 0.0 <= qual["overall_quality_score"] <= 100.0
    assert qual["test_quality"]["total_assertions"] >= 1

    rel = ReleaseIntelligenceEngine(index=idx).evaluate_release(
        proj, version_label="3.7.0", changed_files=["app/core/math_core.py"]
    )
    assert rel["readiness"] in ("READY", "READY_WITH_WARNINGS", "BLOCKED")
    assert "# Release Readiness Report" in rel["release_report_md"]


def test_lazy_hub_master_controller_and_v37_api_endpoints(tmp_path: Path):
    proj = _create_multi_layer_project(tmp_path)
    hub = ProjectIntelligenceHub(db_path=tmp_path / "pi.db", authorized_root=proj)
    assert hub.loaded_engines_count() == 0

    router = V3APIRouter(workspace_root=proj, db_path=tmp_path / "api_v37.db")
    headers = {"X-Test": "1"}
    claims = {"sub": "arch_1", "role": "user"}

    # Backward compatibility: v3.0 health + v3.5 registry (0 agents loaded)
    hc, _, hb = router.handle_public("GET", "/api/v3/system/health", headers)
    assert hc == 200 and hb["status"] == "ok"
    rc, _, rb = router.handle_authenticated("GET", "/api/v3/multi-agent/registry", claims=claims, body={}, sec_headers=headers)
    assert rc == 200 and rb["loaded_agents"] == 0

    # MasterController integration with pi_op
    mc, _, mb = router.handle_authenticated(
        "POST",
        "/api/v3/multi-agent/execute",
        claims=claims,
        body={"objective": "Check release readiness", "params": {"pi_op": "release"}},
        sec_headers=headers,
    )
    assert mc == 200 and mb["status"] == "COMPLETED"

    # v3.7 REST API endpoints
    ac, _, ab = router.handle_authenticated("POST", "/api/v3/project-intel/analyze", claims=claims, body={}, sec_headers=headers)
    assert ac == 200 and ab["version"] == "3.7.0" and ab["architecture"]["components_count"] >= 3

    ic, _, ib = router.handle_authenticated(
        "POST", "/api/v3/project-intel/impact", claims=claims, body={"changed_files": ["app/core/math_core.py"]}, sec_headers=headers
    )
    assert ic == 200 and "app/service/processor.py" in ib["downstream_impacted_files"]

    # Project isolation check (403 on path traversal)
    fc, _, fb = router.handle_authenticated(
        "POST", "/api/v3/project-intel/analyze", claims=claims, body={"project_root": "../outside"}, sec_headers=headers
    )
    assert fc == 403 and fb["code"] == "PROJECT_ISOLATION_VIOLATION"

    # Project memory API
    pmc, _, pmb = router.handle_authenticated(
        "POST",
        "/api/v3/project-intel/memory",
        claims=claims,
        body={"record_type": "ADR", "title": "Incremental AST", "content": "Cache AST by mtime+sha256."},
        sec_headers=headers,
    )
    assert pmc == 200 and pmb["version"] == 1
    gmc, _, gmb = router.handle_authenticated("GET", "/api/v3/project-intel/memory", claims=claims, body={}, sec_headers=headers)
    assert gmc == 200 and len(gmb["records"]) == 1
