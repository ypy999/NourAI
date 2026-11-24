"""Jebrex AI — Phase F Production Training Gate, Corpus Auditor & External Checkpoint Sync.

Enforces strict scientific and operational safety gates BEFORE launching any
production 125M or 350M pretraining run:
1. Audits exact per-language (AR / EN / FR) document, byte, ByteTokenizer token,
   and MultilingualBPETokenizer token counts across train/validation/test splits.
2. Verifies dataset SHA-256 manifest integrity (`verify_versioned_dataset`).
3. HARD BLOCKS production training when `overall_production_readiness_gate != "DATA_READY"`,
   when cross-split contamination > 0, or when required CUDA GPU VRAM is unavailable.
4. Provides verifiable external storage export/import (SHA-256 portable bundles)
   and validates cloud notebook cell syntax cleanly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from data.versioning import verify_versioned_dataset
from model.jebrex_transformer import verify_preset_parameter_counts
from tokenizer.bpe_tokenizer import MultilingualBPETokenizer
from tokenizer.byte_tokenizer import ByteTokenizer


class ProductionGateBlockedError(RuntimeError):
    """Raised when production training is attempted without passing all mandatory gates."""

    def __init__(self, blockers: Sequence[str], report: Dict[str, Any]) -> None:
        self.blockers = list(blockers)
        self.report = report
        super().__init__("Production training BLOCKED by readiness gate: " + "; ".join(blockers))


def audit_multilingual_corpus_tokens(
    splits_dir: Path = Path("data/processed/jebrex_v1_clean/splits"),
    bpe_path: Path = Path("tokenizer/artifacts/jebrex_bpe_v1.json"),
) -> Dict[str, Any]:
    """Measure exact per-language (ar/en/fr) corpus sizes and token counts (Byte & BPE)."""
    byte_tok = ByteTokenizer()
    bpe_tok = MultilingualBPETokenizer.load(bpe_path)
    splits_dir = Path(splits_dir)

    split_summary: Dict[str, Any] = {}
    lang_totals: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "chars": 0, "utf8_bytes": 0, "byte_tokens": 0, "bpe_tokens": 0}
    )
    overall = {"documents": 0, "chars": 0, "utf8_bytes": 0, "byte_tokens": 0, "bpe_tokens": 0}

    for split_name in ("train", "validation", "test"):
        fp = splits_dir / f"{split_name}.jsonl"
        by_lang: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"documents": 0, "chars": 0, "utf8_bytes": 0, "byte_tokens": 0, "bpe_tokens": 0}
        )
        split_tot = {"documents": 0, "chars": 0, "utf8_bytes": 0, "byte_tokens": 0, "bpe_tokens": 0}
        if fp.is_file():
            with fp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = str(obj.get("text", ""))
                    lang = str(obj.get("language", "unknown")).strip().lower()
                    if not text:
                        continue
                    n_chars = len(text)
                    n_bytes = len(text.encode("utf-8"))
                    n_byte_tok = len(byte_tok.encode(text, add_bos=True, add_eos=True))
                    n_bpe_tok = len(bpe_tok.encode(text, add_bos=True, add_eos=True))

                    for bucket in (by_lang[lang], split_tot, lang_totals[lang], overall):
                        bucket["documents"] += 1
                        bucket["chars"] += n_chars
                        bucket["utf8_bytes"] += n_bytes
                        bucket["byte_tokens"] += n_byte_tok
                        bucket["bpe_tokens"] += n_bpe_tok

        split_summary[split_name] = {
            "path": str(fp).replace("\\", "/"),
            "size_bytes": fp.stat().st_size if fp.is_file() else 0,
            "totals": split_tot,
            "by_language": dict(sorted(by_lang.items())),
        }

    return {
        "audited_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits_dir": str(splits_dir).replace("\\", "/"),
        "bpe_tokenizer_path": str(bpe_path).replace("\\", "/"),
        "bpe_vocab_size": bpe_tok.vocab_size,
        "overall": overall,
        "by_language_all_splits": dict(sorted(lang_totals.items())),
        "splits": split_summary,
    }


def verify_cloud_notebook_executable(
    notebook_path: Path = Path("notebooks/JebrexAI_Cloud_GPU_Training.ipynb"),
) -> Dict[str, Any]:
    """Validate notebook JSON structure and Python AST syntax of every code cell (skipping shell/magic lines)."""
    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    errors: List[str] = []
    for idx, cell in enumerate(code_cells, start=1):
        raw_lines = cell.get("source", [])
        if isinstance(raw_lines, str):
            raw_lines = raw_lines.splitlines(keepends=True)
        # Replace IPython line magics / shell escapes (`!git`, `%cd`) with valid no-ops for AST verification
        sanitized = [
            "pass  # ipython_magic\n" if ln.lstrip().startswith(("!", "%")) else ln
            for ln in raw_lines
        ]
        src = "".join(sanitized)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            errors.append(f"cell_{idx}: {exc}")

    return {
        "notebook_path": str(notebook_path).replace("\\", "/"),
        "total_cells": len(cells),
        "code_cells": len(code_cells),
        "syntax_valid": len(errors) == 0,
        "errors": errors,
    }


class ExternalCheckpointStorage:
    """Portable SHA-256 bundle sync adapter for Google Drive / GCS / Cloudflare R2 (outside Git)."""

    def __init__(self, remote_root: Path | str) -> None:
        self.remote_root = Path(remote_root)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def push_portable_bundle(self, local_bundle_dir: Path, bundle_name: str) -> Dict[str, Any]:
        """Copy a SHA-256 portable bundle to mounted external storage (e.g. Google Drive) and verify integrity."""
        src = Path(local_bundle_dir)
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        dst = self.remote_root / bundle_name
        dst.mkdir(parents=True, exist_ok=True)

        verified_files: List[Dict[str, Any]] = []
        for fentry in manifest.get("files", []):
            fname = fentry["name"]
            expected_sha = fentry["sha256"]
            src_file = src / fname
            dst_file = dst / fname
            tmp_file = dst / (fname + ".tmp")
            shutil.copy2(src_file, tmp_file)
            actual_sha = self._sha256(tmp_file)
            if actual_sha != expected_sha:
                tmp_file.unlink(missing_ok=True)
                raise ValueError(f"External sync SHA-256 mismatch for {fname}: {actual_sha} != {expected_sha}")
            tmp_file.replace(dst_file)
            verified_files.append({"name": fname, "sha256": actual_sha, "size_bytes": dst_file.stat().st_size})

        shutil.copy2(src / "manifest.json", dst / "manifest.json")
        return {
            "pushed_to": str(dst).replace("\\", "/"),
            "verified_files": verified_files,
            "sha256_verified": True,
        }

    def pull_portable_bundle(self, bundle_name: str, local_target_dir: Path) -> Dict[str, Any]:
        """Pull a portable bundle from mounted external storage and verify SHA-256 before returning."""
        src = self.remote_root / bundle_name
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        dst = Path(local_target_dir)
        dst.mkdir(parents=True, exist_ok=True)

        for fentry in manifest.get("files", []):
            fname = fentry["name"]
            expected_sha = fentry["sha256"]
            tmp_file = dst / (fname + ".tmp")
            shutil.copy2(src / fname, tmp_file)
            actual_sha = self._sha256(tmp_file)
            if actual_sha != expected_sha:
                tmp_file.unlink(missing_ok=True)
                raise ValueError(f"Pulled bundle SHA-256 mismatch for {fname}")
            tmp_file.replace(dst / fname)

        shutil.copy2(src / "manifest.json", dst / "manifest.json")
        return {
            "pulled_from": str(src).replace("\\", "/"),
            "local_bundle_dir": str(dst).replace("\\", "/"),
            "sha256_verified": True,
        }


def evaluate_production_training_gate(
    config_path: Path = Path("configs/training_125m.json"),
    dataset_manifest_path: Path = Path("data/manifests/jebrex_v1_clean.v1.0.0.manifest.json"),
    bpe_path: Path = Path("tokenizer/artifacts/jebrex_bpe_v1.json"),
    notebook_path: Path = Path("notebooks/JebrexAI_Cloud_GPU_Training.ipynb"),
    report_out: Optional[Path] = Path("reports/production_execution/production_training_gate_check.json"),
) -> Dict[str, Any]:
    """Evaluate all production readiness gates strictly; never returns READY unless every gate passes."""
    train_spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ds_manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    ds_integrity = verify_versioned_dataset(dataset_manifest_path)
    corpus_audit = audit_multilingual_corpus_tokens(
        splits_dir=Path(ds_manifest["splits"]["train"]["path"]).parent,
        bpe_path=bpe_path,
    )
    preset_audit = verify_preset_parameter_counts()
    nb_check = verify_cloud_notebook_executable(notebook_path)

    blockers: List[str] = []

    if not ds_integrity.get("valid", False):
        blockers.append("Dataset SHA-256 manifest verification failed.")

    contam = int(ds_manifest.get("contamination", {}).get("total_cross_split_contamination", 1))
    if contam != 0:
        blockers.append(f"Cross-split data contamination detected ({contam}).")

    prod_gate = ds_manifest.get("overall_production_readiness_gate", "DATA_NOT_READY")
    if train_spec.get("requires_production_data_gate", True) and prod_gate != "DATA_READY":
        blockers.append(
            f"Dataset production gate is {prod_gate} (need >= {train_spec['min_production_tokens_total']:,} licensed tokens; "
            f"current train split has {corpus_audit['splits']['train']['totals']['bpe_tokens']:,} BPE tokens / "
            f"{corpus_audit['splits']['train']['totals']['byte_tokens']:,} Byte tokens)."
        )

    bpe_vocab = int(corpus_audit["bpe_vocab_size"])
    target_vocab = int(train_spec.get("target_tokenizer_vocab_size", 32768))
    if bpe_vocab < target_vocab:
        blockers.append(
            f"Tokenizer vocabulary ({bpe_vocab:,}) has not reached production target ({target_vocab:,}) due to small local corpus."
        )

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    gpu_vram_gib = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_ok else 0.0
    )
    min_vram = float(train_spec.get("min_gpu_vram_gib", 15.0))
    if train_spec.get("requires_cuda_gpu", True):
        if not cuda_ok:
            blockers.append(f"CUDA GPU unavailable on current host (required >= {min_vram} GiB VRAM for {train_spec['preset']}).")
        elif gpu_vram_gib < min_vram:
            blockers.append(f"GPU VRAM ({gpu_vram_gib} GiB) below required minimum ({min_vram} GiB) for {train_spec['preset']}.")

    if not nb_check["syntax_valid"]:
        blockers.append(f"Cloud notebook syntax validation failed: {nb_check['errors']}")

    gate_status = "PRODUCTION_TRAINING_READY" if not blockers else "PRODUCTION_TRAINING_BLOCKED"

    report = {
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_path).replace("\\", "/"),
        "target_stage": train_spec["stage"],
        "target_preset": train_spec["preset"],
        "expected_parameters": train_spec["expected_parameters"],
        "verified_meta_parameters": preset_audit[train_spec["preset"]]["total_parameters"],
        "gate_status": gate_status,
        "training_allowed": len(blockers) == 0,
        "blockers": blockers,
        "hardware": {
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "gpu_vram_gib": gpu_vram_gib,
            "required_vram_gib": min_vram,
        },
        "dataset_integrity": ds_integrity,
        "dataset_production_gate": prod_gate,
        "corpus_token_audit": corpus_audit,
        "notebook_check": nb_check,
        "preset_parameter_audit": {
            k: {
                "total_parameters": v["total_parameters"],
                "parameters_millions": v["parameters_millions"],
                "weights_bf16_mib": v["weights_bf16_mib"],
                "adamw_training_states_fp32_mib": v["adamw_training_states_fp32_mib"],
            }
            for k, v in preset_audit.items()
        },
    }
    if report_out is not None:
        out_p = Path(report_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def launch_production_training(
    config_path: Path = Path("configs/training_125m.json"),
    dataset_manifest_path: Path = Path("data/manifests/jebrex_v1_clean.v1.0.0.manifest.json"),
    bpe_path: Path = Path("tokenizer/artifacts/jebrex_bpe_v1.json"),
) -> Dict[str, Any]:
    """Strict production launcher: ALWAYS blocks if any production gate fails."""
    report = evaluate_production_training_gate(
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        bpe_path=bpe_path,
    )
    if not report["training_allowed"]:
        raise ProductionGateBlockedError(report["blockers"], report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Jebrex AI Production Training Gate & Preflight")
    parser.add_argument("--config", type=Path, default=Path("configs/training_125m.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/jebrex_v1_clean.v1.0.0.manifest.json"))
    parser.add_argument("--bpe", type=Path, default=Path("tokenizer/artifacts/jebrex_bpe_v1.json"))
    parser.add_argument("--strict-launch", action="store_true", help="Exit non-zero if production gate blocks training.")
    args = parser.parse_args(argv)

    rep = evaluate_production_training_gate(
        config_path=args.config,
        dataset_manifest_path=args.manifest,
        bpe_path=args.bpe,
    )
    print(
        json.dumps(
            {
                "gate_status": rep["gate_status"],
                "target_preset": rep["target_preset"],
                "verified_meta_parameters": rep["verified_meta_parameters"],
                "dataset_production_gate": rep["dataset_production_gate"],
                "train_bpe_tokens": rep["corpus_token_audit"]["splits"]["train"]["totals"]["bpe_tokens"],
                "train_by_lang_bpe": {
                    k: v["bpe_tokens"]
                    for k, v in rep["corpus_token_audit"]["splits"]["train"]["by_language"].items()
                },
                "notebook_syntax_valid": rep["notebook_check"]["syntax_valid"],
                "blockers_count": len(rep["blockers"]),
                "blockers": rep["blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict_launch and not rep["training_allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
