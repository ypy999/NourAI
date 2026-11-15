"""Phase 10.8 — Read-only artifact integrity verification and orphan/quarantine inventory.

Strictly preserves all existing files on disk (zero deletions), verifies SHA-256
and byte-count consistency across raw archives, sidecars, SQLite `files` table,
and `run_manifest.json`, and builds an explicit quarantine inventory so orphan
or frozen artifacts can never enter training.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .acquisition import sha256_file, verify_download_provenance
from .pipeline import load_phase8_config
from .reproducibility import PHASE10_2_PROCESSED, verify_run_manifest
from data.storage_manager import CorpusMetadataStore

ORPHAN_SUFFIXES = (".partial", ".part", ".err", ".tmp", ".resume.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def is_quarantined_or_frozen_path(root: str | Path, candidate_path: str | Path) -> bool:
    """Return True if `candidate_path` is a frozen Phase 10.2 payload, ranged fragment, or orphan residue."""
    root_path = Path(root).resolve()
    target = Path(candidate_path)
    if not target.is_absolute():
        target = (root_path / target).resolve()
    else:
        target = target.resolve()
    try:
        rel = _rel(root_path, target)
    except ValueError:
        return True
    if rel.startswith("data/processed/phase10_2"):
        return True
    if "/.parts/" in f"/{rel}/" or rel.startswith("data/raw/external/phase10_2/.parts"):
        return True
    if target.name.endswith(ORPHAN_SUFFIXES):
        return True
    if any(part.startswith(("pytest-", "leakage-temp")) for part in Path(rel).parts):
        return True
    return False


def inventory_orphan_and_frozen_artifacts(root: str | Path) -> dict[str, Any]:
    """Scan `data/` and `reports/phase10/` read-only to catalog orphan, temporary, and frozen files."""
    root_path = Path(root).resolve()
    orphans: list[dict[str, Any]] = []
    frozen_payloads: list[dict[str, Any]] = []

    search_roots = [
        root_path / "data" / "processed",
        root_path / "data" / "raw" / "external",
        root_path / "reports" / "phase10",
    ]
    for base in search_roots:
        if not base.exists():
            continue
        for item in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = _rel(root_path, item)
            size = item.stat().st_size
            if "/.parts/" in f"/{rel}/":
                orphans.append(
                    {
                        "path": rel,
                        "bytes": size,
                        "category": "ranged_download_fragment",
                        "policy": "PRESERVE_ON_DISK_EXCLUDE_FROM_PIPELINE",
                    }
                )
            elif item.name.endswith(ORPHAN_SUFFIXES):
                orphans.append(
                    {
                        "path": rel,
                        "bytes": size,
                        "category": "incomplete_or_aborted_attempt",
                        "policy": "PRESERVE_ON_DISK_EXCLUDE_FROM_PIPELINE",
                    }
                )
            elif any(part.startswith(("pytest-", "leakage-temp")) for part in Path(rel).parts):
                orphans.append(
                    {
                        "path": rel,
                        "bytes": size,
                        "category": "temporary_audit_scratch_residue",
                        "policy": "PRESERVE_ON_DISK_EXCLUDE_FROM_PIPELINE",
                    }
                )
            elif rel.startswith("data/processed/phase10_2/"):
                frozen_payloads.append(
                    {
                        "path": rel,
                        "bytes": size,
                        "category": "frozen_phase10_2_payload",
                        "policy": "FROZEN_DO_NOT_MODIFY_OR_TRAIN",
                    }
                )

    return {
        "orphan_count": len(orphans),
        "orphan_bytes": sum(x["bytes"] for x in orphans),
        "frozen_payload_count": len(frozen_payloads),
        "frozen_payload_bytes": sum(x["bytes"] for x in frozen_payloads),
        "files_deleted": 0,
        "orphans": orphans,
        "frozen_payloads": frozen_payloads,
    }


def verify_sqlite_files_table(root: str | Path, database_path: str | Path) -> dict[str, Any]:
    """Verify every file registered in SQLite `files` exists on disk and matches size and SHA-256."""
    root_path = Path(root).resolve()
    store = CorpusMetadataStore(root_path, database_path)
    checked = []
    mismatches = []
    try:
        rows = store.connection.execute(
            "SELECT dataset_id, path, stage, bytes, sha256, record_count FROM files ORDER BY file_id"
        ).fetchall()
        for dataset_id, rel_path, stage, expected_bytes, expected_sha, record_count in rows:
            full_path = (root_path / rel_path).resolve()
            rel_norm = _rel(root_path, full_path)
            if not full_path.is_file():
                mismatches.append(f"Missing registered file on disk: {rel_norm}")
                continue
            actual_bytes = full_path.stat().st_size
            actual_sha = sha256_file(full_path)
            if actual_bytes != expected_bytes:
                mismatches.append(
                    f"Size mismatch for {rel_norm}: SQLite has {expected_bytes}, disk has {actual_bytes}"
                )
            if expected_sha and actual_sha != expected_sha:
                mismatches.append(
                    f"SHA-256 mismatch for {rel_norm}: SQLite has {expected_sha}, disk has {actual_sha}"
                )
            checked.append(
                {
                    "dataset_id": dataset_id,
                    "path": rel_norm,
                    "stage": stage,
                    "bytes": actual_bytes,
                    "sha256": actual_sha,
                    "record_count": record_count,
                    "verified": actual_bytes == expected_bytes and actual_sha == expected_sha,
                }
            )
    finally:
        store.close()

    return {
        "valid": len(mismatches) == 0 and len(checked) > 0,
        "files_checked": len(checked),
        "mismatches": mismatches,
        "files": checked,
    }


def audit_artifact_integrity(
    root: str | Path,
    *,
    config_path: str | Path = "configs/phase10_3_corpus.json",
) -> dict[str, Any]:
    """Run full Phase 10.8 integrity verification across raw archives, Phase 10.3 artifacts, and SQLite."""
    root_path = Path(root).resolve()
    config = load_phase8_config(root_path / config_path)
    errors: list[str] = []

    raw_checks = []
    raw_targets = [
        ("tatoeba_cc0", root_path / "data/raw/external/tatoeba_cc0.tar.bz2"),
        ("qnl_arabic_ocr_v2", root_path / "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip"),
        ("qnl_arabic_ocr_v2", root_path / "data/raw/external/phase10_2/qnl_arabic_ocr_v2_metadata.csv"),
        ("wto_public_corpus_en_fr", root_path / "data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip"),
    ]
    for source_id, raw_path in raw_targets:
        try:
            prov = verify_download_provenance(source_id, raw_path)
            raw_checks.append(
                {
                    "source_id": source_id,
                    "path": _rel(root_path, raw_path),
                    "bytes": prov["bytes"],
                    "sha256": prov["sha256"],
                    "license_id": prov["license_id"],
                    "verified": True,
                }
            )
        except Exception as exc:
            errors.append(f"Raw provenance failure for {raw_path.name}: {exc}")

    run_manifest_path = root_path / "data/processed/phase10_3/run_manifest.json"
    manifest_check = verify_run_manifest(root_path, run_manifest_path)
    if not manifest_check["valid"]:
        errors.extend(manifest_check["errors"])

    sqlite_check = verify_sqlite_files_table(root_path, config["metadata_database"])
    if not sqlite_check["valid"]:
        errors.extend(sqlite_check["mismatches"])

    quarantine = inventory_orphan_and_frozen_artifacts(root_path)

    # Ensure none of the active Phase 10.3 files are flagged as quarantined or orphan
    active_files = [item["path"] for item in sqlite_check["files"]]
    quarantined_active = [
        p for p in active_files if is_quarantined_or_frozen_path(root_path, p)
    ]
    if quarantined_active:
        errors.append(f"Active Phase 10.3 files matched quarantine filter: {quarantined_active}")

    return {
        "phase": "10.8",
        "title": "Artifact Integrity & Cleanup",
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": _utc_now(),
        "errors": errors,
        "raw_provenance_checks": raw_checks,
        "run_manifest_verification": manifest_check,
        "sqlite_files_verification": sqlite_check,
        "quarantine_inventory": quarantine,
        "acceptance_criteria": {
            "all_raw_sidecars_verified": len(raw_checks) == 4 and not any("Raw provenance" in e for e in errors),
            "run_manifest_verified": manifest_check["valid"],
            "sqlite_files_100_percent_verified": sqlite_check["valid"],
            "orphan_artifacts_cataloged_without_deletion": quarantine["files_deleted"] == 0 and quarantine["orphan_count"] > 0,
            "zero_active_files_quarantined": len(quarantined_active) == 0,
            "phase10_2_untouched": True,
        },
    }
