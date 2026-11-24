"""Phase A — Jebrex AI Production Data Versioning & Readiness Blueprint.

Creates immutable, SHA-256 versioned dataset manifests under `data/manifests/`
and enforces strict separation between pipeline-validation datasets and
billion-token production datasets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from data.jebrex_data_gate import audit_dataset_splits


PRODUCTION_SCALE_TARGETS: dict[str, dict[str, Any]] = {
    "level_1_working_lm": {
        "target_tokens_total": 3_000_000_000,
        "target_tokens_per_lang": {"ar": 1_000_000_000, "en": 1_000_000_000, "fr": 1_000_000_000},
        "storage_tier": "GCS / Cloudflare R2 (Parquet/JSONL zstd shards)",
        "recommended_licensed_sources": [
            {"id": "wikimedia_ar_en_fr", "license": "CC-BY-SA-4.0 (attribution + share-alike review required)"},
            {"id": "tatoeba_cc0", "license": "CC0-1.0 (verified)"},
            {"id": "wto_public_corpus_en_fr", "license": "WTO Public Terms (verified, attribution required)"},
            {"id": "qnl_arabic_ocr_v2_filtered", "license": "Public Domain + CC0 metadata (strict OCR filter required)"},
            {"id": "open_books_cc_by_public_domain", "license": "CC0 / Public Domain / CC-BY-4.0 (per-source audit required)"},
        ],
    },
    "level_2_instruction_chat": {
        "target_pretrain_tokens": 50_000_000_000,
        "target_sft_conversations": 100_000,
        "target_dpo_preference_pairs": 30_000,
        "human_review_requirement": "Mandatory human quality verification for Arabic, English, and French; never label synthetic templates as human dialogue.",
    },
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def create_versioned_dataset_manifest(
    splits_dir: str | Path,
    *,
    dataset_id: str,
    version: str,
    output_dir: str | Path = "data/manifests",
) -> dict[str, Any]:
    """Audit splits, hash every split file, and emit an immutable versioned dataset manifest."""
    sdir = Path(splits_dir)
    audit = audit_dataset_splits(sdir, dataset_name=f"{dataset_id}@{version}")

    split_files: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        fp = sdir / f"{split_name}.jsonl"
        if fp.is_file():
            split_files[split_name] = {
                "path": fp.as_posix(),
                "size_bytes": fp.stat().st_size,
                "sha256": _sha256_file(fp),
                "line_count": sum(1 for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()),
            }

    manifest_fingerprint_src = json.dumps(
        {k: v["sha256"] for k, v in sorted(split_files.items())},
        sort_keys=True,
    )
    dataset_fingerprint = hashlib.sha256(manifest_fingerprint_src.encode("utf-8")).hexdigest()

    # Compute exact gap ratio to Level 1 production target per language
    lang_gaps: dict[str, Any] = {}
    for lg in ("ar", "en", "fr"):
        actual_toks = audit["languages"][lg]["splits"]["train"]["byte_tokens"]
        target_toks = PRODUCTION_SCALE_TARGETS["level_1_working_lm"]["target_tokens_per_lang"][lg]
        lang_gaps[lg] = {
            "actual_train_tokens": actual_toks,
            "level_1_target_tokens": target_toks,
            "coverage_pct": round(100.0 * actual_toks / target_toks, 8),
            "pipeline_validation_gate": audit["languages"][lg]["pipeline_validation_gate"],
            "production_readiness_gate": audit["languages"][lg]["general_llm_production_gate"],
        }

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": version,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": split_files,
        "contamination": audit["contamination"],
        "overall_pipeline_validation_gate": audit["overall_pipeline_validation_gate"],
        "overall_production_readiness_gate": audit["overall_general_llm_production_gate"],
        "per_language_readiness_and_gap": lang_gaps,
        "production_scale_targets": PRODUCTION_SCALE_TARGETS,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{dataset_id}.{version}.manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_versioned_dataset(manifest_path: str | Path) -> dict[str, Any]:
    """Verify split files match immutable SHA-256 digests recorded in a dataset manifest."""
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for split_name, meta in manifest.get("splits", {}).items():
        fp = Path(meta["path"])
        if not fp.is_file():
            mismatches.append(f"missing:{split_name}:{meta['path']}")
        elif _sha256_file(fp) != meta["sha256"]:
            mismatches.append(f"sha256_mismatch:{split_name}:{meta['path']}")
    return {
        "valid": len(mismatches) == 0,
        "dataset_id": manifest.get("dataset_id"),
        "version": manifest.get("version"),
        "dataset_fingerprint_sha256": manifest.get("dataset_fingerprint_sha256"),
        "mismatches": mismatches,
    }


def run_phase_a_data_versioning() -> dict[str, Any]:
    m_clean = create_versioned_dataset_manifest(
        "data/processed/jebrex_v1_clean/splits",
        dataset_id="jebrex_v1_clean",
        version="v1.0.0",
    )
    m_sft = create_versioned_dataset_manifest(
        "data/processed/jebrex_v1_clean_sft/splits",
        dataset_id="jebrex_v1_clean_sft",
        version="v1.0.0",
    )
    return {
        "jebrex_v1_clean": {
            "fingerprint": m_clean["dataset_fingerprint_sha256"],
            "pipeline_gate": m_clean["overall_pipeline_validation_gate"],
            "production_gate": m_clean["overall_production_readiness_gate"],
        },
        "jebrex_v1_clean_sft": {
            "fingerprint": m_sft["dataset_fingerprint_sha256"],
            "pipeline_gate": m_sft["overall_pipeline_validation_gate"],
            "production_gate": m_sft["overall_production_readiness_gate"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_phase_a_data_versioning(), indent=2, ensure_ascii=False))
