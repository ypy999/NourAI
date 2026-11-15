"""CLI & utility script to verify, activate, or roll back NourAI releases (v1.0.0, v1.1.0, v1.2.0)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from inference.engine import GenerationConfig, InferenceEngine


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activate_or_rollback_release(
    target_version: str,
    *,
    releases_root: str | Path = "releases",
    active_pointer_path: str | Path = "releases/active_release.json",
) -> dict[str, Any]:
    """Verify SHA-256 checksum of target_version checkpoint, load it into InferenceEngine, and set active pointer."""
    rel_dir = Path(releases_root) / target_version
    manifest_path = rel_dir / "release_manifest.json"
    if not manifest_path.is_file():
        manifest_path = rel_dir / "update_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"release manifest not found for {target_version}: {rel_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "model_checkpoint" in manifest:
        ckpt_path = Path(manifest["model_checkpoint"]["path"])
        expected_sha = manifest["model_checkpoint"]["sha256"]
    elif "artifacts" in manifest and "aligned_model_checkpoint" in manifest["artifacts"]:
        ckpt_path = Path(manifest["artifacts"]["aligned_model_checkpoint"]["path"])
        expected_sha = manifest["artifacts"]["aligned_model_checkpoint"]["sha256"]
    elif "checkpoint" in manifest:
        ckpt_path = Path(manifest["checkpoint"])
        expected_sha = manifest["checkpoint_sha256"]
    else:
        raise ValueError(f"unable to locate checkpoint entry in {manifest_path}")

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint missing on disk: {ckpt_path}")

    actual_sha = sha256_file(ckpt_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {target_version}: expected {expected_sha}, got {actual_sha}")

    # Load and run smoke generation to prove checkpoint is healthy
    engine = InferenceEngine.from_checkpoint(ckpt_path)
    smoke = engine.generate("مرحبا NourAI", GenerationConfig(max_new_tokens=6, strategy="greedy"))

    pointer = {
        "active_version": target_version,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": actual_sha,
        "verified_parameters": engine.model.count_parameters(),
        "smoke_tokens_generated": smoke["generated_token_count"],
        "switched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_ptr = Path(active_pointer_path)
    out_ptr.parent.mkdir(parents=True, exist_ok=True)
    out_ptr.write_text(json.dumps(pointer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate or roll back a NourAI release version.")
    parser.add_argument("--version", required=True, choices=["v1.0.0", "v1.1.0", "v1.2.0"])
    args = parser.parse_args()
    result = activate_or_rollback_release(args.version)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
