from pathlib import Path

from corpus.certification import certify_phase10


def test_phase10_12_final_certification_all_gates_pass(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cert = certify_phase10(root, cache_dir=tmp_path)
    assert cert["status"] == "PASS", cert["failed_gates"]
    assert cert["certification_verdict"] == "CERTIFIED_FOR_PHASE_11_ARCHITECTURE_AND_INITIAL_TRAINING"
    assert cert["reproduction_plan_ready"] is True
    assert cert["failed_gates"] == []
    assert all(g["status"] == "PASS" for g in cert["gates"].values())
    assert all(cert["constraints_verified"].values())
