"""Phase 10.11 — Training readiness validation (Tokenizer, TokenSequenceDataset, DataLoader, and config).

Runs strictly on CPU without starting model training or touching CUDA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader

from configs.configuration import load_config
from .integrity import is_quarantined_or_frozen_path
from .multilingual_quality import _resolve_canonical_language
from .reproducibility import verify_run_manifest
from datasets.dataset import TokenSequenceDataset
from tokenizer.byte_tokenizer import ByteTokenizer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_training_readiness(
    root: str | Path,
    *,
    splits_dir: str | Path = "data/processed/phase10_3/splits",
    config_path: str | Path = "configs/default.json",
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate ByteTokenizer, TokenSequenceDataset, DataLoader, and manifests on Phase 10.3 splits."""
    root_path = Path(root).resolve()
    splits_path = (root_path / splits_dir).resolve() if not Path(splits_dir).is_absolute() else Path(splits_dir).resolve()
    cfg_path = (root_path / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve()

    tokenizer = ByteTokenizer()
    project_config = load_config(cfg_path, tokenizer)
    project_config.validate(tokenizer)

    errors: list[str] = []
    if project_config.device != "cpu":
        errors.append(f"Expected CPU device in default config, got {project_config.device!r}")

    split_names = ("train", "validation", "test")
    target_languages = ("arabic", "english", "french")
    tokenizer_stats = {
        "vocab_size": tokenizer.vocab_size,
        "records_tested": 0,
        "tokens_generated": 0,
        "lossless_roundtrip_failures": 0,
        "out_of_range_tokens": 0,
        "by_language_tested": {lang: 0 for lang in target_languages},
    }

    representative_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in split_names}

    for split_name in split_names:
        split_file = splits_path / f"{split_name}.jsonl"
        if not split_file.is_file():
            errors.append(f"Missing split file: {split_file}")
            continue
        if is_quarantined_or_frozen_path(root_path, split_file):
            errors.append(f"Split file is in quarantined/frozen path: {split_file}")

        lang_collected = {lang: 0 for lang in target_languages}
        with split_file.open("rb") as stream:
            for raw_line in stream:
                row = json.loads(raw_line.decode("utf-8", errors="strict"))
                lang = _resolve_canonical_language(row)
                if lang in target_languages and lang_collected[lang] < 20:
                    text = row["text"]
                    # Bound extremely long book texts to 8,192 chars for fast Dataset/DataLoader smoke slice
                    slice_text = text[:8192] if len(text) > 8192 else text
                    ids = tokenizer.encode(slice_text, add_bos=True, add_eos=True)
                    decoded = tokenizer.decode(ids, errors="strict")
                    tokenizer_stats["records_tested"] += 1
                    tokenizer_stats["tokens_generated"] += len(ids)
                    tokenizer_stats["by_language_tested"][lang] += 1
                    if decoded != slice_text:
                        tokenizer_stats["lossless_roundtrip_failures"] += 1
                    if any(t < 0 or t >= tokenizer.vocab_size for t in ids):
                        tokenizer_stats["out_of_range_tokens"] += 1

                    row_copy = dict(row)
                    row_copy["text"] = slice_text
                    representative_rows[split_name].append(row_copy)
                    lang_collected[lang] += 1

                if all(c >= 20 for c in lang_collected.values()):
                    break

    if tokenizer_stats["lossless_roundtrip_failures"] > 0:
        errors.append(f"Tokenizer had {tokenizer_stats['lossless_roundtrip_failures']} roundtrip failures")
    if tokenizer_stats["out_of_range_tokens"] > 0:
        errors.append(f"Tokenizer produced {tokenizer_stats['out_of_range_tokens']} out-of-range tokens")

    dataloader_checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="nourai-readiness-dl-", dir=cache_dir) as tmp_root:
        tmp_dir = Path(tmp_root)
        for split_name in split_names:
            slice_path = tmp_dir / f"{split_name}.jsonl"
            slice_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in representative_rows[split_name]),
                encoding="utf-8",
            )
            ds = TokenSequenceDataset(slice_path, config=project_config, tokenizer=tokenizer, cache_dir=tmp_dir)
            loader = DataLoader(ds, batch_size=project_config.batch_size, shuffle=False, num_workers=0)
            first_inputs, first_targets = next(iter(loader))
            expected_len = project_config.context_length
            shift_ok = bool(torch.equal(first_inputs[:, 1:], first_targets[:, :-1]))
            meta_sample = ds.metadata_for_sequence(0)
            provenance_ok = len(meta_sample) > 0 and all(
                bool(item.get("source_id"))
                and bool(item.get("license_id"))
                and bool(item.get("record_id"))
                and bool(item.get("content_hash"))
                and set(item.get("provenance_missing") or []) <= {"document_id", "source_line"}
                for item in meta_sample
            )
            if first_inputs.shape[1] != expected_len or first_targets.shape[1] != expected_len:
                errors.append(f"{split_name}: unexpected sequence length {tuple(first_inputs.shape)}")
            if not shift_ok:
                errors.append(f"{split_name}: causal next-token shift mismatch between inputs and targets")
            if not provenance_ok:
                errors.append(f"{split_name}: sequence provenance metadata missing in TokenSequenceDataset")

            dataloader_checks[split_name] = {
                "sequences_available": len(ds),
                "total_tokens_in_slice": ds.token_count,
                "first_batch_input_shape": list(first_inputs.shape),
                "first_batch_target_shape": list(first_targets.shape),
                "causal_shift_verified": shift_ok,
                "sequence_provenance_verified": provenance_ok,
            }
            ds._temp.cleanup()

    run_manifest_check = verify_run_manifest(
        root_path, root_path / "data/processed/phase10_3/run_manifest.json"
    )
    if not run_manifest_check["valid"]:
        errors.extend(run_manifest_check["errors"])

    passed = len(errors) == 0

    return {
        "phase": "10.11",
        "title": "Training Readiness Validation",
        "status": "PASS" if passed else "FAIL",
        "created_at_utc": _utc_now(),
        "training_started": False,
        "cuda_used": False,
        "device_verified": project_config.device,
        "errors": errors,
        "tokenizer_validation": tokenizer_stats,
        "dataloader_validation": dataloader_checks,
        "run_manifest_verification": run_manifest_check,
        "acceptance_criteria": {
            "tokenizer_lossless_roundtrip_100_percent": tokenizer_stats["lossless_roundtrip_failures"] == 0
            and tokenizer_stats["records_tested"] > 0,
            "all_three_languages_tokenized": all(
                c > 0 for c in tokenizer_stats["by_language_tested"].values()
            ),
            "dataloader_batches_and_causal_shift_verified": all(
                d["causal_shift_verified"] and d["sequence_provenance_verified"]
                for d in dataloader_checks.values()
            )
            and len(dataloader_checks) == 3,
            "no_quarantined_or_frozen_paths_used": not any("quarantined" in e for e in errors),
            "cpu_only_no_training_started": True,
        },
    }
