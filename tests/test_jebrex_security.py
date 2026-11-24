"""Phase 9 acceptance tests — 12-domain Jebrex AI security audit and critical/high remediation verification."""
from __future__ import annotations

from pathlib import Path

from api.security_performance import run_phase20_security_and_performance_audit
from security.audit import run_stage_i_security_verification
from security.security_audit_v2 import run_comprehensive_security_audit


def test_comprehensive_12_domain_security_audit_passes_with_zero_critical_or_high(tmp_path: Path):
    rep = run_comprehensive_security_audit(report_dir=tmp_path)
    assert rep["security_gate"] == "PASS_NO_OPEN_CRITICAL_OR_HIGH"
    assert rep["all_verification_checks_passed"] is True
    assert rep["open_counts"]["CRITICAL"] == 0
    assert rep["open_counts"]["HIGH"] == 0
    for check_name, passed in rep["checks"].items():
        assert passed is True, f"Security check failed: {check_name}"
    assert (tmp_path / "security_audit_report.json").is_file()
    assert (tmp_path / "security_audit_report.md").is_file()


def test_legacy_stage_i_and_phase20_security_suites_pass():
    st_i = run_stage_i_security_verification()
    assert st_i["all_gates_passed"] is True
    assert st_i["status"] == "SECURITY_VERIFIED"

    p20 = run_phase20_security_and_performance_audit()
    assert p20["all_checks_passed"] is True
    assert p20["status"] == "CERTIFIED_FOR_PHASE_21_DEPLOYMENT_AND_PRODUCTION"
