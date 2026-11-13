"""Dataset provenance and processing manifests without guessed metadata."""
from datetime import date
import json
import os
from pathlib import Path
import tempfile

from .tokenizer_analysis import BYTE_TOKENIZER_VERSION


def _is_supplied(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in {"unknown", "not provided", "n/a", "none"}
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_supplied(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(_is_supplied(item) for item in value.values())
    return True


def training_readiness(manifest):
    """Return explicit eligibility and blockers; no license is inferred."""
    blockers = []
    if not _is_supplied(manifest.get("source")):
        blockers.append("source is missing or unverified")
    if not _is_supplied(manifest.get("license")):
        blockers.append("license is missing or unverified")
    if manifest.get("record_count", 0) <= 0:
        blockers.append("dataset contains no accepted records")
    splits = manifest.get("split_statistics") or {}
    if not splits:
        blockers.append("train/validation/test split statistics are missing")
    else:
        for split in ("train", "validation", "test"):
            if splits.get(split, 0) <= 0:
                blockers.append(f"{split} split is empty")
    return {"eligible": not blockers, "blockers": blockers}


def build_manifest(*, dataset_name, dataset_version, source=None, license=None,
                   collection_date=None, language_statistics=None, quality_statistics=None,
                   tokenizer_statistics=None, split_statistics=None, pipeline_statistics=None,
                   cleaning_config=None, deduplication_results=None, split_ratios=None,
                   random_seed=None, processed_files=(), processing_status="processed",
                   warnings=None, errors=None):
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError("dataset_version must be a non-empty string")
    if collection_date is not None and not isinstance(collection_date, str):
        raise ValueError("collection_date must be an explicitly supplied ISO date string or None")
    if collection_date is not None:
        try:
            date.fromisoformat(collection_date)
        except ValueError as exc:
            raise ValueError("collection_date must use ISO YYYY-MM-DD") from exc
    file_list = [Path(path) for path in processed_files]
    file_size = 0
    for path in file_list:
        if path.exists() and path.is_file():
            file_size += path.stat().st_size
    pipeline = pipeline_statistics or {}
    split_stats = split_statistics or {}
    if hasattr(split_stats, "split_counts"):
        split_stats = split_stats.split_counts
    elif isinstance(split_stats, dict) and "split_counts" in split_stats:
        split_stats = split_stats["split_counts"]
    if hasattr(pipeline, "as_dict"):
        pipeline = pipeline.as_dict()
    if hasattr(language_statistics, "as_dict"):
        language_statistics = language_statistics.as_dict()
    if hasattr(quality_statistics, "as_dict"):
        quality_statistics = quality_statistics.as_dict()
    if hasattr(tokenizer_statistics, "as_dict"):
        tokenizer_statistics = tokenizer_statistics.as_dict()
    record_count = pipeline.get("accepted_records", pipeline.get("unique_records"))
    if record_count is None:
        record_count = sum(split_stats.get(name, 0) for name in ("train", "validation", "test")) if split_stats else 0
    character_count = pipeline.get("output_characters")
    if character_count is None:
        character_count = (language_statistics or {}).get("total_characters", 0)
    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name.strip(),
        "dataset_version": dataset_version.strip(),
        "source": source,
        "license": license,
        "collection_date": collection_date,
        "language_distribution": language_statistics,
        "record_count": record_count,
        "character_count": character_count,
        "token_count": (tokenizer_statistics or {}).get("tokens", 0),
        "file_size_bytes": file_size,
        "files": [str(path) for path in file_list],
        "cleaning_configuration": cleaning_config,
        "deduplication_results": deduplication_results,
        "quality_statistics": quality_statistics,
        "tokenizer_statistics": tokenizer_statistics,
        "split_statistics": split_stats,
        "split_ratios": split_ratios,
        "random_seed": random_seed,
        "tokenizer_version": (tokenizer_statistics or {}).get("tokenizer_version", BYTE_TOKENIZER_VERSION),
        "processing_status": processing_status,
        "warnings": list(warnings or []),
        "errors": list(errors or []),
    }
    readiness = training_readiness(manifest)
    manifest["training_eligible"] = readiness["eligible"]
    manifest["training_blockers"] = readiness["blockers"]
    if not readiness["eligible"]:
        manifest["processing_status"] = "blocked_missing_provenance_or_data"
        manifest["warnings"].extend(readiness["blockers"])
    return manifest


def write_manifest(manifest, path):
    """Write a new UTF-8 JSON manifest atomically, never overwriting a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise FileExistsError(f"manifest already exists: {path}")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path
