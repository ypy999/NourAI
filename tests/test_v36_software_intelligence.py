"""JABRIX AI v3.6.0 — Comprehensive Autonomous Software Intelligence Test Suite.

Covers all 10 modules, project isolation, zero host execution of untrusted code,
human approval workflow, deterministic patch rollback, lazy loading, 8-agent integration,
and v3.6 REST API endpoints with v3.0-v3.5 backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from nourai.api.v3_routes import V3APIRouter
from nourai.software_intel import (
    ApprovalManager,
    AuthorizedProjectScope,
    BugDetectionEngine,
    CodeIntelligenceEngine,
    CodeOptimizerEngine,
    DependencyAnalyzer,
    DocumentationEngine,
    PatchGeneratorEngine,
    ProjectIsolationError,
    ProjectUnderstandingEngine,
    RegressionGuard,
    SoftwareIntelligenceHub,
    TestGeneratorEngine,
)


def _create_sample_project(root: Path) -> Path:
    proj = root / "sample_proj"
    (proj / "app").mkdir(parents=True)
    (proj / "tests").mkdir(parents=True)
    (proj / "docs").mkdir(parents=True)

    (proj / "pyproject.toml").write_text('[project]\nname = "sample"\nrequires-python = ">=3.10"\n', encoding="utf-8")
    (proj / "requirements.txt").write_text("requests==2.31.0\nnumpy\n", encoding="utf-8")
    (proj / "requirements.lock").write_text("requests==2.32.0\n", encoding="utf-8")
    (proj / "docs" / "README.md").write_text("# Sample Project\n", encoding="utf-8")

    (proj / "app" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "app" / "mod_a.py").write_text(
        "import app.mod_b\n\n"
        "def compute_total(items, acc=[]):\n"
        "    out = ''\n"
        "    for x in items:\n"
        "        for y in items:\n"
        "            out += str(x + y)\n"
        "    return out\n",
        encoding="utf-8",
    )
    (proj / "app" / "mod_b.py").write_text(
        "from app.mod_a import compute_total\n"
        "import subprocess\n\n"
        "API_KEY = 'super_secret_token_value_12345'\n\n"
        "def run_helper(cmd):\n"
        "    try:\n"
        "        return eval(cmd)\n"
        "    except:\n"
        "        subprocess.run(cmd, shell=True)\n"
        "        return None\n"
        "        x = 1\n",
        encoding="utf-8",
    )
    (proj / "tests" / "test_app.py").write_text(
        "from app.mod_a import compute_total\n\n"
        "def test_compute_total():\n"
        "    assert compute_total([1]) == '2'\n",
        encoding="utf-8",
    )
    return proj


def test_project_understanding_and_code_intelligence(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    pu = ProjectUnderstandingEngine().analyze(proj)
    assert pu["total_files_scanned"] >= 7
    assert pu["file_classifications"]["test"] == 1
    assert pu["file_classifications"]["docs"] == 1
    assert "app.mod_a" in pu["dependency_map"]

    ci = CodeIntelligenceEngine().index_project(proj)
    sym_names = {s["name"] for s in ci["symbols"]}
    assert {"compute_total", "run_helper", "test_compute_total"} <= sym_names
    assert "compute_total" in ci["cross_file_references"]


def test_dependency_analyzer_cycles_and_version_conflicts(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    dep = DependencyAnalyzer().analyze(proj)
    assert dep["python_requires"] == ">=3.10"
    assert dep["has_circular_dependencies"] is True
    assert any("app.mod_a" in c and "app.mod_b" in c for c in dep["circular_dependencies"])
    issue_types = {i["type"] for i in dep["compatibility_issues"]}
    assert "conflicting_version_specs" in issue_types
    assert "unpinned_dependency" in issue_types


def test_bug_detection_and_code_optimizer_zero_execution(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    bugs = BugDetectionEngine().scan_project(proj)
    rules = {f["rule_id"] for f in bugs["findings"]}
    assert "SEC001_HARDCODED_SECRET" in rules
    assert "SEC002_DYNAMIC_EXEC" in rules
    assert "SEC004_SUBPROCESS_SHELL" in rules
    assert "BUG001_BARE_EXCEPT" in rules
    assert "BUG002_MUTABLE_DEFAULT" in rules
    assert "BUG003_UNREACHABLE_CODE" in rules

    opt = CodeOptimizerEngine().analyze_project(proj)
    cats = {s["category"] for s in opt["suggestions"]}
    assert "bottleneck_nested_loops" in cats
    assert "string_concat_loop" in cats


def test_test_generator_and_documentation_engine(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    tg = TestGeneratorEngine()
    cov = tg.analyze_coverage(proj)
    assert cov["public_symbols_total"] >= 2
    assert cov["covered_symbols_count"] >= 1
    assert cov["uncovered_symbols_count"] >= 1

    unit = tg.generate_unit_tests(proj, "app/mod_a.py")
    assert "def test_compute_total_basic" in unit["generated_code"]

    bugs = BugDetectionEngine().scan_project(proj)
    reg = tg.generate_regression_tests(bugs["findings"])
    assert reg["tests_generated"] >= 1

    pu = ProjectUnderstandingEngine().analyze(proj)
    dep = DependencyAnalyzer().analyze(proj)
    arch_md = DocumentationEngine.generate_architecture_doc(pu)
    dep_md = DocumentationEngine.generate_dependency_doc(dep)
    assert "```mermaid" in arch_md
    assert "Circular Dependency Cycles" in dep_md


def test_patch_generator_regression_guard_approval_and_rollback(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    db_path = tmp_path / "ledger.db"
    orig_file = proj / "app" / "mod_a.py"
    orig_text = orig_file.read_text(encoding="utf-8")

    # 1. Invalid syntax patch is rejected
    bad_patch = PatchGeneratorEngine().generate_patch(proj, "app/mod_a.py", "def broken(:\n")
    assert bad_patch["validation"]["valid"] is False
    assert orig_file.read_text(encoding="utf-8") == orig_text

    # 2. Breaking API change caught by RegressionGuard
    guard = RegressionGuard()
    brk = guard.check_patch("app/mod_a.py", orig_text, "def renamed_func(x):\n    return x\n")
    assert brk["api_compatible"] is False and brk["passed"] is False

    # 3. Removed test caught by RegressionGuard
    test_orig = (proj / "tests" / "test_app.py").read_text(encoding="utf-8")
    rm_test = guard.check_patch("tests/test_app.py", test_orig, "# deleted tests\n")
    assert rm_test["tests_preserved"] is False and rm_test["passed"] is False

    # 4. Valid compatible patch: propose -> refuse unapproved apply -> approve -> apply -> rollback
    good_proposed = (
        "import app.mod_b\n\n"
        "def compute_total(items, acc=None):\n"
        "    parts = [str(x + y) for x in items for y in items]\n"
        "    return ''.join(parts)\n"
    )
    patch = PatchGeneratorEngine().generate_patch(proj, "app/mod_a.py", good_proposed, rationale="Fix mutable default")
    assert patch["validation"]["valid"] is True
    assert patch["original_file_modified"] is False
    assert guard.check_patch("app/mod_a.py", orig_text, good_proposed)["passed"] is True
    assert guard.run_isolated_syntax_check(proj, "app/mod_a.py", good_proposed)["compile_ok"] is True

    mgr = ApprovalManager(db_path)
    prop = mgr.propose_patch(user_id="dev1", patch_record=patch)
    pid = prop["patch_id"]
    assert prop["status"] == "PENDING_APPROVAL"
    # Original file must STILL be untouched
    assert orig_file.read_text(encoding="utf-8") == orig_text

    # Refuses apply without approval
    with pytest.raises(PermissionError):
        mgr.apply_approved_patch(pid)

    # Approve & apply
    mgr.approve_patch(pid, reviewer="lead_engineer")
    applied = mgr.apply_approved_patch(pid)
    assert applied["status"] == "APPLIED"
    assert "acc=None" in orig_file.read_text(encoding="utf-8")

    # Rollback restores exact original bytes & SHA-256
    rb = mgr.rollback_patch(pid)
    assert rb["status"] == "ROLLED_BACK"
    assert orig_file.read_text(encoding="utf-8") == orig_text
    assert len(mgr.list_patches()) == 1


def test_project_isolation_blocks_path_traversal(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    outside = tmp_path / "secret_outside.py"
    outside.write_text("SECRET = 'DO_NOT_READ'\n", encoding="utf-8")

    scope = AuthorizedProjectScope(proj)
    with pytest.raises(ProjectIsolationError):
        scope.resolve_safe("../secret_outside.py")
    with pytest.raises(ProjectIsolationError):
        scope.resolve_safe(outside)


def test_lazy_loading_and_eight_agent_api_integration(tmp_path: Path):
    proj = _create_sample_project(tmp_path)
    hub = SoftwareIntelligenceHub(db_path=tmp_path / "hub.db", authorized_root=proj)
    assert hub.loaded_engines_count() == 0

    router = V3APIRouter(workspace_root=proj, db_path=tmp_path / "api_v36.db")
    headers = {"X-Test": "1"}
    claims = {"sub": "eng_1", "role": "user"}

    # Backward compatibility: v3.0 health + v3.5 registry
    hc, _, hb = router.handle_public("GET", "/api/v3/system/health", headers)
    assert hc == 200 and hb["status"] == "ok"
    rc, _, rb = router.handle_authenticated("GET", "/api/v3/multi-agent/registry", claims=claims, body={}, sec_headers=headers)
    assert rc == 200 and rb["total_agents"] == 8 and rb["loaded_agents"] == 0

    # Multi-agent coding + research integration on project
    mc, _, mb = router.handle_authenticated(
        "POST",
        "/api/v3/multi-agent/execute",
        claims=claims,
        body={"objective": "Scan project for static bugs", "params": {"sw_op": "bugs"}},
        sec_headers=headers,
    )
    assert mc == 200 and mb["status"] == "COMPLETED"

    # v3.6 REST API: analyze + traversal block (403) + propose/approve/rollback
    ac, _, ab = router.handle_authenticated("POST", "/api/v3/software/analyze", claims=claims, body={}, sec_headers=headers)
    assert ac == 200 and ab["version"] == "3.6.0" and ab["bug_detection"]["total_findings"] >= 1

    fc, _, fb = router.handle_authenticated(
        "POST",
        "/api/v3/software/patches/propose",
        claims=claims,
        body={"target_file": "../escape.py", "proposed_content": "x = 1\n"},
        sec_headers=headers,
    )
    assert fc == 403 and fb["code"] == "PROJECT_ISOLATION_VIOLATION"

    pc, _, pb = router.handle_authenticated(
        "POST",
        "/api/v3/software/patches/propose",
        claims=claims,
        body={
            "target_file": "app/mod_a.py",
            "proposed_content": "import app.mod_b\n\ndef compute_total(items, acc=None):\n    return ''\n",
            "rationale": "Safe refactor",
        },
        sec_headers=headers,
    )
    assert pc == 200 and pb["proposal"]["status"] == "PENDING_APPROVAL"
    pid = pb["proposal"]["patch_id"]

    apc, _, apb = router.handle_authenticated(
        "POST", "/api/v3/software/patches/approve", claims=claims, body={"patch_id": pid, "apply": True}, sec_headers=headers
    )
    assert apc == 200 and apb["applied"]["status"] == "APPLIED"

    rbc, _, rbb = router.handle_authenticated(
        "POST", "/api/v3/software/patches/rollback", claims=claims, body={"patch_id": pid}, sec_headers=headers
    )
    assert rbc == 200 and rbb["status"] == "ROLLED_BACK"
