from pathlib import Path

from corpus.training_readiness import validate_training_readiness


def test_phase10_11_training_readiness_passes_on_cpu(tmp_path):
    root = Path(__file__).resolve().parents[1]
    res = validate_training_readiness(root, cache_dir=tmp_path)
    assert res["status"] == "PASS", res["errors"]
    assert res["training_started"] is False
    assert res["cuda_used"] is False
    assert res["device_verified"] == "cpu"
    assert all(res["acceptance_criteria"].values())
    for split_name in ("train", "validation", "test"):
        assert res["dataloader_validation"][split_name]["causal_shift_verified"] is True
        assert res["dataloader_validation"][split_name]["sequence_provenance_verified"] is True
