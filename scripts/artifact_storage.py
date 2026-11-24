"""Cloud-First Artifact Manifest, SHA-256 Integrity, Lazy Cached Fetcher, Retention Policy & Storage Audit.

Keeps Git lean by indexing external/binary weights and datasets via SHA-256 manifests
without ever deleting original data, checkpoints, or reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.request import Request, urlopen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact_manifest(
    root: str | Path = ".",
    *,
    output_path: str | Path = "manifests/artifacts_manifest.json",
) -> dict[str, Any]:
    """Scan checkpoints, releases, tokenizer artifacts, and clean splits to produce a SHA-256 manifest."""
    repo = Path(root).resolve()
    entries: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    duplicate_bytes = 0
    duplicates: list[dict[str, Any]] = []

    patterns = [
        ("checkpoint", "checkpoints/**/*.pt"),
        ("release_weight", "releases/**/*.pt"),
        ("tokenizer_artifact", "tokenizer/artifacts/*.json"),
        ("clean_dataset", "data/processed/jebrex_v1_clean/splits/*.jsonl"),
        ("clean_sft", "data/processed/jebrex_v1_clean_sft/splits/*.jsonl"),
    ]

    for kind, pat in patterns:
        for path in sorted(repo.glob(pat)):
            if not path.is_file():
                continue
            rel = path.relative_to(repo).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            if digest in seen_hashes:
                duplicate_bytes += size
                duplicates.append({"path": rel, "duplicate_of": seen_hashes[digest], "size_bytes": size, "sha256": digest})
            else:
                seen_hashes[digest] = rel
            entries.append(
                {
                    "path": rel,
                    "kind": kind,
                    "size_bytes": size,
                    "size_mib": round(size / (1024 * 1024), 4),
                    "sha256": digest,
                    "cloud_uri": None,
                    "storage_tier": "local_workspace_indexed",
                }
            )

    total_bytes = sum(e["size_bytes"] for e in entries)
    disk = shutil.disk_usage(repo)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_count": len(entries),
        "total_indexed_mib": round(total_bytes / (1024 * 1024), 4),
        "duplicate_bytes": duplicate_bytes,
        "duplicate_mib": round(duplicate_bytes / (1024 * 1024), 4),
        "duplicates_detected": duplicates,
        "local_disk_free_gib": round(disk.free / (1024 ** 3), 3),
        "retention_policy": {
            "always_preserve_last_healthy_checkpoint": True,
            "max_history_snapshots_per_run": 2,
            "never_delete_original_datasets": True,
        },
        "artifacts": entries,
    }

    out = repo / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_artifact_manifest(
    manifest_path: str | Path = "manifests/artifacts_manifest.json",
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Verify every locally present artifact matches its recorded SHA-256 digest."""
    repo = Path(root).resolve()
    manifest = json.loads((repo / manifest_path).read_text(encoding="utf-8"))
    verified = 0
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []

    for item in manifest.get("artifacts", []):
        p = repo / item["path"]
        if not p.is_file():
            missing.append(item["path"])
            continue
        actual = sha256_file(p)
        if actual != item["sha256"]:
            mismatched.append({"path": item["path"], "expected": item["sha256"], "actual": actual})
        else:
            verified += 1

    return {
        "valid": len(missing) == 0 and len(mismatched) == 0,
        "verified_count": verified,
        "missing": missing,
        "mismatched": mismatched,
    }


def fetch_cached_artifact(
    name: str,
    *,
    url: str,
    expected_sha256: str,
    cache_dir: str | Path = "data/cache",
    timeout: float = 30.0,
) -> Path:
    """Lazy-download an external cloud artifact into local cache and verify SHA-256 atomically."""
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(name).name
    target = cdir / safe_name
    if target.is_file() and sha256_file(target) == expected_sha256:
        return target

    tmp = cdir / f".{safe_name}.{int(time.time_ns())}.part"
    try:
        req = Request(url, headers={"User-Agent": "JebrexArtifactFetcher/1.0"})
        with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        actual = sha256_file(tmp)
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {safe_name}: expected {expected_sha256}, got {actual}")
        tmp.replace(target)
        return target
    finally:
        if tmp.exists():
            tmp.unlink()


def plan_checkpoint_retention(
    history_dir: str | Path,
    *,
    keep: int = 2,
    healthy_validator=None,
) -> dict[str, Any]:
    """Compute a safe retention plan (dry-run by default) that NEVER drops the last healthy checkpoint."""
    hdir = Path(history_dir)
    if not hdir.is_dir():
        return {"kept": [], "prunable": [], "protected_healthy": None}
    files = sorted(hdir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"kept": [], "prunable": [], "protected_healthy": None}

    healthy_files = [f for f in files if (healthy_validator(f) if healthy_validator else f.stat().st_size > 0)]
    protected = healthy_files[0] if healthy_files else files[0]

    kept = list(files[: max(1, int(keep))])
    if protected not in kept:
        kept.append(protected)
    kept_set = set(kept)
    prunable = [f for f in files if f not in kept_set]
    return {
        "kept": [str(p) for p in kept],
        "prunable": [str(p) for p in prunable],
        "protected_healthy": str(protected),
    }


def generate_monthly_storage_report(
    root: str | Path = ".",
    *,
    output_json: str | Path = "reports/production_execution/storage_audit_snapshot.json",
) -> dict[str, Any]:
    repo = Path(root).resolve()
    manifest = build_artifact_manifest(repo)
    verification = verify_artifact_manifest(root=repo)

    dir_sizes: dict[str, float] = {}
    for child in sorted(repo.iterdir()):
        if child.name in (".git", "__pycache__", ".pytest_cache"):
            continue
        if child.is_dir():
            sz = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            dir_sizes[child.name] = round(sz / (1024 * 1024), 3)

    report = {
        "month": time.strftime("%Y-%m", time.gmtime()),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace_dir_sizes_mib": dir_sizes,
        "indexed_artifacts_mib": manifest["total_indexed_mib"],
        "duplicate_artifacts_mib": manifest["duplicate_mib"],
        "duplicates_detected": manifest["duplicates_detected"],
        "manifest_verification": verification,
        "local_disk_free_gib": manifest["local_disk_free_gib"],
        "monthly_cloud_cost_usd_actual": 0.0,
        "active_free_tiers": {
            "github": "Free Public Repo + Actions",
            "firebase_hosting": "Spark Plan (10 GB storage, 360 MB/day transfer, 24/7 CDN)",
            "render_api": "Not active (HTTP 404 probed 2026-11-24; Free plan sleeps after 15m idle)",
        },
    }
    out = repo / output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jebrex AI Cloud Storage & Artifact Manifest Tool")
    parser.add_argument("--report", action="store_true", help="Generate artifact manifest and monthly storage audit")
    args = parser.parse_args()
    rep = generate_monthly_storage_report()
    print(json.dumps({
        "indexed_artifacts_mib": rep["indexed_artifacts_mib"],
        "duplicate_artifacts_mib": rep["duplicate_artifacts_mib"],
        "manifest_valid": rep["manifest_verification"]["valid"],
        "verified_count": rep["manifest_verification"]["verified_count"],
        "local_disk_free_gib": rep["local_disk_free_gib"],
    }, indent=2))
