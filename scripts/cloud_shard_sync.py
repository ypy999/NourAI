"""Jebrex AI — Phase G Batched Cloud Storage Shard Sync & Checkpoint Retention Guard.

Provides storage-efficient, SHA-256-verified dataset sharding and batched streaming
transfers for Cloudflare R2 (recommended: $0 egress), Google Cloud Storage (GCS),
and Google Drive — keeping all large corpora and checkpoints strictly outside GitHub.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence
from urllib.request import Request, urlopen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_sharded_dataset(
    source_jsonl: Path,
    out_shards_dir: Path,
    *,
    dataset_id: str,
    split_name: str,
    records_per_shard: int = 5000,
) -> Dict[str, Any]:
    """Partition a JSONL split into SHA-256-indexed shards suitable for batched cloud download."""
    source_jsonl = Path(source_jsonl)
    out_shards_dir = Path(out_shards_dir)
    out_shards_dir.mkdir(parents=True, exist_ok=True)
    if records_per_shard < 1:
        raise ValueError("records_per_shard must be >= 1")

    shards: List[Dict[str, Any]] = []
    shard_idx = 0
    buf: List[str] = []

    def _flush() -> None:
        nonlocal shard_idx
        if not buf:
            return
        fname = f"{split_name}_shard_{shard_idx:05d}.jsonl"
        fp = out_shards_dir / fname
        fp.write_text("".join(buf), encoding="utf-8")
        shards.append(
            {
                "shard_index": shard_idx,
                "filename": fname,
                "record_count": len(buf),
                "size_bytes": fp.stat().st_size,
                "sha256": sha256_file(fp),
                "uri": fp.resolve().as_uri(),
            }
        )
        shard_idx += 1
        buf.clear()

    with source_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            buf.append(line if line.endswith("\n") else (line + "\n"))
            if len(buf) >= records_per_shard:
                _flush()
    _flush()

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "split": split_name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shard_count": len(shards),
        "total_records": sum(s["record_count"] for s in shards),
        "total_bytes": sum(s["size_bytes"] for s in shards),
        "shards": shards,
    }
    manifest_path = out_shards_dir / f"{split_name}_shards_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class BatchedCloudShardFetcher:
    """Streams dataset shards in bounded batches with strict SHA-256 verification & cache eviction."""

    def __init__(
        self,
        manifest: Dict[str, Any],
        cache_dir: Path,
        *,
        batch_size: int = 2,
        evict_after_batch: bool = True,
        timeout_sec: int = 60,
    ) -> None:
        self.manifest = manifest
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, int(batch_size))
        self.evict_after_batch = bool(evict_after_batch)
        self.timeout_sec = timeout_sec
        self.peak_cache_bytes: int = 0
        self.verified_shards_count: int = 0

    def _download_one_verified(self, shard: Dict[str, Any]) -> Path:
        dst = self.cache_dir / shard["filename"]
        expected_sha = shard["sha256"]
        if dst.is_file() and sha256_file(dst) == expected_sha:
            self.verified_shards_count += 1
            return dst

        tmp = dst.with_suffix(dst.suffix + ".tmp")
        uri = shard["uri"]
        req = Request(uri, headers={"User-Agent": "JebrexAI-ShardFetcher/1.0"}) if uri.startswith("http") else uri
        h = hashlib.sha256()
        with urlopen(req, timeout=self.timeout_sec) as resp, tmp.open("wb") as out_f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                h.update(chunk)
        actual_sha = h.hexdigest()
        if actual_sha != expected_sha:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Shard SHA-256 mismatch for {shard['filename']}: got {actual_sha}, expected {expected_sha}"
            )
        tmp.replace(dst)
        self.verified_shards_count += 1
        return dst

    def iter_verified_batches(self, start_shard: int = 0) -> Iterator[List[Path]]:
        """Yield batches of SHA-256 verified local shard paths, evicting finished batches to bound disk usage."""
        shards = self.manifest.get("shards", [])[start_shard:]
        for i in range(0, len(shards), self.batch_size):
            batch_meta = shards[i : i + self.batch_size]
            local_paths = [self._download_one_verified(s) for s in batch_meta]
            current_bytes = sum(p.stat().st_size for p in self.cache_dir.glob("*") if p.is_file())
            self.peak_cache_bytes = max(self.peak_cache_bytes, current_bytes)
            try:
                yield local_paths
            finally:
                if self.evict_after_batch:
                    for p in local_paths:
                        p.unlink(missing_ok=True)


def enforce_healthy_checkpoint_retention(
    checkpoint_dir: Path,
    *,
    keep_last: int = 2,
    health_check: Optional[Callable[[Path], bool]] = None,
) -> Dict[str, Any]:
    """Prune old checkpoints while GUARANTEED NEVER deleting the last verified healthy checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    validator = health_check or (lambda p: p.is_file() and p.stat().st_size > 0)
    candidates = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not candidates:
        return {"kept": [], "deleted": [], "protected_last_healthy": None}

    healthy = [p for p in candidates if validator(p)]
    last_healthy = healthy[-1] if healthy else None

    keep_set = set(candidates[-max(1, keep_last) :])
    if last_healthy is not None:
        keep_set.add(last_healthy)

    deleted: List[str] = []
    for p in candidates:
        if p not in keep_set:
            p.unlink(missing_ok=True)
            deleted.append(str(p).replace("\\", "/"))

    return {
        "kept": [str(p).replace("\\", "/") for p in candidates if p in keep_set],
        "deleted": deleted,
        "protected_last_healthy": str(last_healthy).replace("\\", "/") if last_healthy else None,
    }


def build_cloud_transfer_commands(provider: str, remote_bucket: str, local_dir: str) -> Dict[str, str]:
    """Generate secret-free CLI sync commands for Cloudflare R2, GCS, or Google Drive."""
    p = provider.strip().lower()
    if p == "r2":
        return {
            "provider": "Cloudflare R2 (S3-compatible, $0 egress)",
            "upload": f"aws s3 sync {local_dir} s3://{remote_bucket} --endpoint-url https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com",
            "download": f"aws s3 sync s3://{remote_bucket} {local_dir} --endpoint-url https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com",
        }
    if p == "gcs":
        return {
            "provider": "Google Cloud Storage (GCS)",
            "upload": f"gcloud storage cp --recursive {local_dir} gs://{remote_bucket}/",
            "download": f"gcloud storage cp --recursive gs://{remote_bucket}/ {local_dir}",
        }
    if p == "gdrive":
        return {
            "provider": "Google Drive (Colab mount / rclone)",
            "upload": f"rclone copy {local_dir} gdrive:{remote_bucket} --checksum",
            "download": f"rclone copy gdrive:{remote_bucket} {local_dir} --checksum",
        }
    raise ValueError(f"Unsupported cloud storage provider: {provider}")
