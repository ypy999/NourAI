"""Phase 2 & Phase A acceptance tests — Cloud storage manifests, SHA-256 verification, retention policy, and data versioning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import pytest

from data.versioning import create_versioned_dataset_manifest, verify_versioned_dataset
from scripts.artifact_storage import (
    build_artifact_manifest,
    fetch_cached_artifact,
    plan_checkpoint_retention,
    verify_artifact_manifest,
)


def test_artifact_manifest_build_and_sha256_verification(tmp_path: Path):
    (tmp_path / "checkpoints" / "demo").mkdir(parents=True)
    ckpt = tmp_path / "checkpoints" / "demo" / "model.pt"
    ckpt.write_bytes(b"jebrex-weights-v1")

    manifest = build_artifact_manifest(tmp_path, output_path="manifests/test_manifest.json")
    assert manifest["artifact_count"] == 1
    ver = verify_artifact_manifest("manifests/test_manifest.json", root=tmp_path)
    assert ver["valid"] is True and ver["verified_count"] == 1

    # Tamper with artifact -> verification MUST fail
    ckpt.write_bytes(b"corrupted-weights")
    ver_bad = verify_artifact_manifest("manifests/test_manifest.json", root=tmp_path)
    assert ver_bad["valid"] is False
    assert len(ver_bad["mismatched"]) == 1


def test_retention_policy_never_prunes_last_healthy_checkpoint(tmp_path: Path):
    hdir = tmp_path / "history"
    hdir.mkdir()
    old_good = hdir / "last-1.pt"
    mid_bad = hdir / "last-2.pt"
    new_bad = hdir / "last-3.pt"
    old_good.write_bytes(b"VALID_CKPT")
    time.sleep(0.02)
    mid_bad.write_bytes(b"")
    time.sleep(0.02)
    new_bad.write_bytes(b"")

    plan = plan_checkpoint_retention(hdir, keep=1, healthy_validator=lambda p: p.stat().st_size > 0)
    assert str(old_good) in plan["kept"]
    assert plan["protected_healthy"] == str(old_good)


def test_lazy_cached_artifact_fetch_verifies_sha256_and_uses_cache(tmp_path: Path):
    src = tmp_path / "remote_file.bin"
    payload = b"cloud-checkpoint-payload-12345"
    src.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    cached = fetch_cached_artifact("model.bin", url=src.resolve().as_uri(), expected_sha256=digest, cache_dir=tmp_path / "cache")
    assert cached.is_file() and cached.read_bytes() == payload

    # Wrong SHA-256 must raise ValueError and leave no corrupt file
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_cached_artifact("bad.bin", url=src.resolve().as_uri(), expected_sha256="0" * 64, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache" / "bad.bin").exists()


def test_versioned_dataset_manifest_verifies_and_reports_production_gap(tmp_path: Path):
    manifest = create_versioned_dataset_manifest(
        "data/processed/jebrex_v1_clean/splits",
        dataset_id="jebrex_v1_clean",
        version="v1.0.0",
        output_dir=tmp_path,
    )
    assert manifest["overall_pipeline_validation_gate"] == "DATA_READY"
    assert manifest["overall_production_readiness_gate"] == "DATA_NOT_READY"
    assert manifest["contamination"]["total_cross_split_contamination"] == 0
    for lg in ("ar", "en", "fr"):
        assert manifest["per_language_readiness_and_gap"][lg]["production_readiness_gate"] == "DATA_NOT_READY"
    ver = verify_versioned_dataset(tmp_path / "jebrex_v1_clean.v1.0.0.manifest.json")
    assert ver["valid"] is True
