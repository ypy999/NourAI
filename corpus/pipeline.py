"""Corpus import and preparation, reusing the Phase 7 processor and splitter."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from data.language_analysis import classify_language
from data.quality import score_record, summarize_quality
from data.splitting import split_sources
from configs.configuration import load_config
from tokenizer.byte_tokenizer import ByteTokenizer

from .importers import iter_source_records
from .acquisition import sha256_file, verify_download_provenance
from data.multilingual import estimate_language
from data.raw_storage import make_raw_record
from data.provenance import canonical_record
from data.source_registry import SourceRegistry
from .licensing import require_approved_source
from .storage import StorageManager


def load_phase8_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "phase8_corpus.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"approved_source_ids", "target_languages", "text_fields", "max_download_bytes",
                "max_json_bytes", "max_zip_member_bytes", "batch_records", "split_ratios",
                "random_seed", "raw_directory", "imported_directory", "splits_directory", "reports_directory",
                "download", "supported_formats", "language_settings", "normalization", "deduplication",
                "quality", "license_policy", "normalized_directory", "approved_directory", "rejected_directory",
                "metadata_database", "report_json", "report_markdown", "dashboard_html"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"phase8 config is missing keys: {', '.join(sorted(missing))}")
    unknown = set(value) - (required | {"schema_version"})
    if unknown:
        raise ValueError(f"unknown phase8 config keys: {', '.join(sorted(unknown))}")
    if value.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for name in ("batch_records", "max_download_bytes", "max_json_bytes", "max_zip_member_bytes", "random_seed"):
        if isinstance(value[name], bool) or not isinstance(value[name], int):
            raise ValueError(f"{name} must be an integer")
    if value["batch_records"] <= 0 or value["max_download_bytes"] <= 0 or value["max_json_bytes"] <= 0 or value["max_zip_member_bytes"] <= 0:
        raise ValueError("batch_records and max_download_bytes must be positive")
    if not isinstance(value["approved_source_ids"], list) or not value["approved_source_ids"]:
        raise ValueError("approved_source_ids must be a nonempty list")
    if not isinstance(value["target_languages"], list) or not value["target_languages"] or any(not isinstance(x, str) or not x for x in value["target_languages"]):
        raise ValueError("target_languages must be a list of language codes")
    if not isinstance(value["text_fields"], list) or not value["text_fields"] or any(not isinstance(x, str) for x in value["text_fields"]):
        raise ValueError("text_fields must be a nonempty list of field names")
    if not isinstance(value["split_ratios"], dict) or set(value["split_ratios"]) != {"train", "validation", "test"}:
        raise ValueError("split_ratios must define train, validation, and test")
    import math
    ratios = list(value["split_ratios"].values())
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("split_ratios must be finite positive values summing to 1")
    for key in ("raw_directory", "imported_directory", "normalized_directory", "splits_directory", "approved_directory",
                "rejected_directory", "reports_directory", "metadata_database", "report_json", "report_markdown", "dashboard_html"):
        relative = Path(value[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{key} must be a project-relative path without parent traversal")
    if not isinstance(value["language_settings"], dict) or set(value["language_settings"]) != set(value["target_languages"]):
        raise ValueError("language_settings must define exactly the target languages")
    if not isinstance(value["supported_formats"], list) or not value["supported_formats"]:
        raise ValueError("supported_formats must be a nonempty list")
    for section in ("download", "normalization", "deduplication", "quality", "license_policy"):
        if not isinstance(value[section], dict):
            raise ValueError(f"{section} settings must be an object")
    for source_id in value["approved_source_ids"]:
        require_approved_source(source_id)
    return value


def import_source_file(source_id, source_path, output_path, *, project_root, languages=None,
                       config=None, resume=False):
    """Resumably stream records into JSONL; raw text and per-record provenance are retained."""
    source = require_approved_source(source_id)
    config = config or load_phase8_config()
    root = Path(project_root).resolve()
    from data.license_manager import LicenseManager
    license_manager = LicenseManager(root / config["metadata_database"])
    license_manager.seed_catalog()
    try:
        source = license_manager.ensure_approved(source_id)
    finally:
        license_manager.close()
    source_path = Path(source_path).resolve()
    provenance = verify_download_provenance(source_id, source_path)
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    storage = StorageManager(root)
    try:
        relative = output_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("import output must remain inside project root") from exc
    if source_path == output_path.resolve():
        raise ValueError("source and imported output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_languages = set(languages or config["target_languages"])
    fingerprint_payload = {"text_fields": config["text_fields"], "max_json_bytes": config["max_json_bytes"],
                          "max_zip_member_bytes": config["max_zip_member_bytes"], "allowed_languages": sorted(allowed_languages)}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    partial = output_path.with_name(output_path.name + ".part")
    checkpoint = output_path.with_name(output_path.name + ".resume.json")
    final_meta = output_path.with_name(output_path.name + ".import.json")
    if output_path.exists():
        if final_meta.exists():
            completed = json.loads(final_meta.read_text(encoding="utf-8"))
            if (completed.get("source_sha256") == provenance["sha256"] and completed.get("output_sha256") == sha256_file(output_path)
                    and completed.get("config_fingerprint") == fingerprint):
                return {**completed, "already_present": True}
        raise FileExistsError(f"refusing to overwrite import output: {output_path}")
    if partial.exists() or checkpoint.exists():
        if not resume:
            raise FileExistsError("an import checkpoint exists; call with resume=True or choose a new output path")
        if not partial.exists() or not checkpoint.exists():
            raise ValueError("resume requires both .part and .resume.json checkpoint files")
    elif resume:
        raise FileNotFoundError("no resumable import checkpoint exists")
    if resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("source_sha256") != provenance["sha256"] or state.get("source_id") != source_id:
            raise ValueError("resume checkpoint belongs to a different source or source version")
        if set(state.get("allowed_languages", [])) != allowed_languages:
            raise ValueError("resume language filter differs from checkpoint")
        if state.get("config_fingerprint") != fingerprint:
            raise ValueError("import configuration differs from checkpoint")
        try:
            with partial.open("r+b") as stream:
                if stream.seek(0, os.SEEK_END) < state["partial_bytes"]:
                    raise ValueError("partial import file is shorter than its checkpoint")
                stream.truncate(state["partial_bytes"])
                stream.seek(state["partial_bytes"])
        except OSError as exc:
            raise ValueError(f"cannot resume partial import: {exc}") from exc
    else:
        state = {"source_id": source_id, "source_sha256": provenance["sha256"],
                 "source_retrieved_at": provenance.get("retrieved_at"), "allowed_languages": sorted(allowed_languages),
                 "source_url": source["url"], "dataset_version": source.get("release_version"),
                 "source_file": str(source_path), "license_id": source["license_id"],
                 "config_fingerprint": fingerprint,
                 "imported_at": datetime.now(timezone.utc).isoformat(), "records_seen": 0, "scanned_items": 0,
                 "accepted_records": 0, "filtered_language_records": 0, "empty_records": 0,
                 "errors": [], "errors_truncated": 0, "languages": {}, "characters": 0, "partial_bytes": 0}
        _write_json_atomic(checkpoint, state)
    stats = state
    counts = Counter(stats.get("languages", {}))
    started = time.perf_counter()
    mode = "ab" if resume else "wb"
    with partial.open(mode) as output:
        for cursor, item in enumerate(iter_source_records(source_path, text_fields=tuple(config["text_fields"]),
                                           max_json_bytes=config["max_json_bytes"],
                                           max_zip_member_bytes=config["max_zip_member_bytes"]), 1):
            if cursor <= int(stats.get("scanned_items", 0)):
                continue
            stats["scanned_items"] = cursor
            if "error" in item:
                if len(stats["errors"]) < 100:
                    stats["errors"].append(item["error"])
                else:
                    stats["errors_truncated"] += 1
            else:
                stats["records_seen"] += 1
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    stats["empty_records"] += 1
                else:
                    lang = item.get("language_code")
                    if allowed_languages and lang is not None and lang not in allowed_languages:
                        stats["filtered_language_records"] += 1
                    else:
                        row = make_raw_record(text, source_id=source_id, dataset_version=source.get("release_version"),
                            license_id=source["license_id"], source_url=source["url"],
                            source_record_id=item.get("source_record_id", item.get("sentence_id")), source_language_code=lang or item.get("language_code"),
                            source_record_number=item.get("record_number"), source_timestamp=item.get("contributor_timestamp"),
                            imported_at=stats["imported_at"])
                        row.update({key: item[key] for key in ("document_id", "sentence_domain", "domain", "domains") if key in item})
                        row["source_name"] = source.get("name")
                        row["license_status"] = source.get("state", source.get("status"))
                        row["source_file"] = str(source_path)
                        row["source_line"] = item.get("record_number")
                        row["text"] = text  # Phase 7 DataPipeline compatibility; this is still raw source text.
                        row["language_estimate"] = estimate_language(text, source_language_code=lang)
                        row = canonical_record(row, source_file=source_path,
                                               source_line=item.get("record_number"),
                                               source_registry=SourceRegistry())
                        output.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                        stats["accepted_records"] += 1
                        stats["characters"] += len(text)
                        counts[lang or classify_language(text)] += 1
            if stats["scanned_items"] % config["batch_records"] == 0:
                output.flush()
                os.fsync(output.fileno())
                stats["languages"] = dict(counts)
                stats["partial_bytes"] = output.tell()
                _write_json_atomic(checkpoint, stats)
        output.flush()
        os.fsync(output.fileno())
        stats["languages"] = dict(counts)
        stats["partial_bytes"] = output.tell()
    stats["processing_seconds"] = time.perf_counter() - started
    stats["imported_file"] = str(output_path)
    stats["imported_bytes"] = partial.stat().st_size
    stats["output_sha256"] = sha256_file(partial)
    stats["ended_at"] = datetime.now(timezone.utc).isoformat()
    if stats["accepted_records"] == 0:
        stats["warning"] = "No records matched the selected language filter; output file is empty."
    if output_path.exists() or final_meta.exists():
        raise FileExistsError("import final output or metadata sidecar appeared concurrently")
    os.replace(partial, output_path)
    _write_json_atomic(final_meta, stats)
    if checkpoint.exists():
        checkpoint.unlink()
    from data.storage_manager import CorpusMetadataStore
    metadata_store = CorpusMetadataStore(root, config["metadata_database"])
    try:
        metadata_store.register_source({"source_id": source_id, **source})
        dataset_id = f"{source_id}:{provenance['sha256'][:12]}"
        metadata_store.register_dataset(dataset_id, source["name"], str(source.get("release_version") or "unversioned"),
            source_id, source["license_id"], status="IMPORTED_LICENSE_VERIFIED", metadata={"imported_at":stats["imported_at"],"source_sha256":provenance["sha256"]})
        metadata_store.register_file(dataset_id, source_path, stage="RAW_ARCHIVE", metadata={"license_id":source["license_id"]})
        metadata_store.register_file(dataset_id, output_path, stage="IMPORTED_RAW_TEXT", record_count=stats["accepted_records"], metadata={"license_id":source["license_id"]})
        indexed = metadata_store.index_jsonl_metadata(dataset_id, output_path, default_license_id=source["license_id"])
        metadata_store.record_operation("import", "COMPLETE", source_id=source_id, details={"records":indexed,"source_sha256":provenance["sha256"]})
        stats["metadata_database"] = str(metadata_store.path)
        stats["metadata_records_indexed"] = indexed
    finally:
        metadata_store.close()
    return stats


def _write_json_atomic(path, payload):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def normalize_imported_file(imported_path, output_path, *, project_root, config=None):
    """Persist traceable raw/normalized/cleaned fields before the Phase 7 splitter."""
    started = time.perf_counter()
    from data.normalization import transform_record
    config = config or load_phase8_config()
    root = Path(project_root).resolve()
    imported_path = Path(imported_path).resolve()
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("normalized output must stay inside project root") from exc
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite normalized output: {output_path}")
    project_config = load_config(root / "configs" / "default.json", ByteTokenizer())
    cleaning = {"remove_control_chars": project_config.cleaning_remove_control_chars,
                "collapse_whitespace": project_config.cleaning_collapse_whitespace,
                "preserve_paragraphs": project_config.cleaning_preserve_paragraphs}
    stats = {"input_file": str(imported_path), "output_file": str(output_path), "records_seen": 0,
             "accepted_records": 0, "rejected_records": 0, "errors": [], "errors_truncated": 0}

    def transformed(*, rejected=False, record_errors=False):
        with imported_path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                try:
                    row = json.loads(raw.decode("utf-8", errors="strict"))
                    if not isinstance(row, dict) or not isinstance(row.get("raw_text", row.get("text")), str):
                        raise ValueError("record requires raw_text/text string")
                    from data.raw_storage import verify_raw_record
                    if "raw_text" in row and not verify_raw_record(row):
                        raise ValueError("raw_text hash or byte count mismatch")
                    result = transform_record(row, unicode_form=config["normalization"]["unicode_form"], cleaning_config=cleaning)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                    if record_errors and len(stats["errors"]) < 100:
                        stats["errors"].append(f"line {line_number}: {exc}")
                    elif record_errors:
                        stats["errors_truncated"] += 1
                    continue
                if not rejected:
                    stats["records_seen"] += 1
                is_rejected = not result["cleaned_text"].strip()
                if is_rejected == rejected:
                    result["record_status"] = "REJECTED_EMPTY_AFTER_CLEANING" if is_rejected else "NORMALIZED_CLEANED"
                    if not rejected:
                        stats["accepted_records"] += 1
                    else:
                        stats["rejected_records"] += 1
                    yield result

    storage = StorageManager(root)
    output = storage.atomic_jsonl(str(relative), transformed(record_errors=True))
    rejected_path = output.with_name(output.stem + ".rejected.jsonl")
    if stats["rejected_records"]:
        storage.atomic_jsonl(str(rejected_path.relative_to(root)), transformed(rejected=True))
        stats["rejected_file"] = str(rejected_path)
    stats["output_bytes"] = output.stat().st_size
    stats["normalization_form"] = config["normalization"]["unicode_form"]
    stats["cleaning_configuration"] = cleaning
    stats["processing_seconds"] = time.perf_counter() - started
    return stats


def prepare_splits(imported_files, *, project_root, config=None, resume=False, checkpoint_path=None):
    """Normalize with trace metadata, then reuse Phase 7 cleaner/deduplicator/splitter."""
    config = config or load_phase8_config()
    root = Path(project_root).resolve()
    output = (root / config["splits_directory"]).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("split output must remain inside project root") from exc
    if output.exists() and not resume:
        raise FileExistsError(f"split directory already exists; select a new run path: {output}")
    normalized_root = (root / config["normalized_directory"]).resolve()
    normalized_root.mkdir(parents=True, exist_ok=True)
    normalized_files = []
    normalization_stats = []
    for imported in imported_files:
        imported = Path(imported).resolve()
        normalized_name = imported.stem + ".normalized.cleaned.jsonl"
        normalized_path = normalized_root / normalized_name
        normalization_stats.append(normalize_imported_file(imported, normalized_path, project_root=root, config=config))
        normalized_files.append(normalized_path)
    project_config = load_config(root / "configs" / "default.json", ByteTokenizer())
    result = split_sources(normalized_files, output, config=project_config,
                         ratios=config["split_ratios"], seed=config["random_seed"],
                         stratify=True, checkpoint_path=checkpoint_path, resume=resume,
                         checkpoint_batch_size=config["batch_records"])
    result.normalized_files = [str(path) for path in normalized_files]
    result.normalization_statistics = normalization_stats
    from data.storage_manager import CorpusMetadataStore
    from data.source_registry import SourceRegistry
    from data.manifest import build_manifest, write_manifest
    from .acquisition import verify_download_provenance
    registry = SourceRegistry()
    source_meta = []
    for imported in imported_files:
        imported_path = Path(imported).resolve()
        meta_path = imported_path.with_name(imported_path.name + ".import.json")
        if not meta_path.is_file():
            raise ValueError(f"import provenance metadata is missing: {meta_path}")
        source_meta.append(json.loads(meta_path.read_text(encoding="utf-8")))
    source_ids = {item.get("source_id") for item in source_meta}
    if len(source_ids) != 1:
        raise ValueError("a Phase 8 run currently accepts one license-verified source at a time")
    source_id = next(iter(source_ids))
    source_entry = registry.get(source_id)
    if not registry.verified_for_ingestion(source_id):
        raise PermissionError(f"source {source_id} is not approved by the active source registry")
    source_license = {"source_id":source_id,"license_id":source_entry["license_id"],"state":"VERIFIED"}
    from data.license_manager import LicenseManager
    license_manager = LicenseManager(root/config["metadata_database"])
    try:
        license_manager.seed_catalog(registry)
        source_license = license_manager.ensure_approved(source_id)
    finally:
        license_manager.close()
    dataset_id = f"{source_entry['source_id']}:phase8"
    store = CorpusMetadataStore(root, config["metadata_database"])
    try:
        store.register_source(source_entry)
        store.register_dataset(dataset_id, source_entry["name"], str(source_entry.get("release_version") or "unversioned"),
                              source_entry["source_id"], source_entry["license_id"], status="CONDITIONALLY_READY",
                              metadata={"unique_records":result.unique_records,"split_counts":result.split_counts})
        for normalized, stat in zip(normalized_files, normalization_stats):
            store.register_file(dataset_id, normalized, stage="NORMALIZED_CLEANED", record_count=stat["accepted_records"], metadata=stat)
        split_paths = {name: output / f"{name}.jsonl" for name in ("train","validation","test")}
        for split, path in split_paths.items():
            file_meta=store.register_file(dataset_id,path,stage="SPLIT_"+split.upper(),record_count=result.split_counts[split],metadata={"seed":result.seed})
            store.index_jsonl_metadata(dataset_id,path,split=split,default_license_id=source_entry["license_id"])
        meta=verify_download_provenance(source_id,Path(source_meta[0]["source_file"]))
        manifest=build_manifest(dataset_name=source_entry["name"],dataset_version=str(source_entry.get("release_version") or meta["sha256"][:12]),
             source={"source_id":source_entry["source_id"],"url":source_entry["url"],"sha256":meta["sha256"]},
             license={"license_id":source_license["license_id"],"state":source_license["state"],"evidence_url":source_entry["source_license_url"]},
             language_statistics={"source_language_codes":source_meta[0].get("languages",{}),"confidence":"source-provided, not independently verified"},
             split_statistics=result.split_counts,pipeline_statistics={"accepted_records":result.unique_records,"duplicates_removed":result.duplicate_records},
             split_ratios=result.requested_ratios,random_seed=result.seed,
             processed_files=[str(path) for path in split_paths.values()],cleaning_config=project_config.to_dict(),
             warnings=list(result.warnings),processing_status="conditionally_ready_for_review")
        manifest["training_eligible"] = False
        manifest["training_blockers"] = ["Arabic validation/test coverage is empty (only 2 Arabic source records).", "The corpus is a sentence subset, not a representative pretraining corpus."]
        manifest["corpus_readiness_status"] = "CONDITIONALLY_READY"
        manifest["source_retrieved_at"] = meta.get("retrieved_at")
        manifest["license_verification_method"] = source_entry.get("verification_method")
        approved_dir=(root/config["approved_directory"])
        approved_dir.mkdir(parents=True,exist_ok=True)
        manifest_path=approved_dir/(source_entry["source_id"]+"_phase8.manifest.json")
        write_manifest(manifest,manifest_path)
        store.record_operation("phase7_process_split", "COMPLETE", source_id=source_entry["source_id"], details={"manifest":str(manifest_path),"split_counts":result.split_counts})
        result.manifest_path=str(manifest_path)
        result.metadata_database=str(store.path)
    finally:
        store.close()
    return result


def corpus_statistics(jsonl_path, *, config=None):
    """Compute streaming language, tokenizer and Phase 7 quality summaries."""
    started = time.perf_counter()
    config = config or load_phase8_config()
    tokenizer = ByteTokenizer()
    language_counts = Counter()
    language_estimate_counts = Counter()
    language_method_counts = Counter()
    confidence_totals = Counter()
    token_count = 0
    chars = 0
    errors = []
    import hashlib
    import sqlite3
    import tempfile
    import unicodedata
    from data.deduplication import DiskDeduplicator

    def scored_records():
        nonlocal token_count, chars
        path = Path(jsonl_path)
        with tempfile.TemporaryDirectory(prefix="nourai-quality-", dir=path.parent) as temp:
            db = sqlite3.connect(str(Path(temp) / "dedup.sqlite3"))
            dedup = DiskDeduplicator(db)
            try:
                with path.open("rb") as stream:
                    for number, raw in enumerate(stream, 1):
                        try:
                            row = json.loads(raw.decode("utf-8", errors="strict"))
                            text = row["text"]
                            if not isinstance(text, str):
                                raise ValueError("text is not a string")
                        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
                            if len(errors) < 100:
                                errors.append(f"{jsonl_path}:{number}: {exc}")
                            continue
                        duplicate_type = dedup.add_if_unique({"text": text, "source": str(path), "line_number": number})
                        if number % config["batch_records"] == 0:
                            db.commit()
                        language_counts[classify_language(text)] += 1
                        estimate = row.get("language_estimate") or estimate_language(text, source_language_code=row.get("source_language_code"))
                        language_estimate_counts[estimate["language"]] += 1
                        language_method_counts[estimate["method"]] += 1
                        confidence_totals[estimate["language"]] += estimate["confidence"]
                        token_count += len(tokenizer.encode(text, add_eos=True))
                        chars += len(text)
                        yield score_record(text, duplicate=duplicate_type is not None)
            finally:
                db.close()
    quality = summarize_quality(scored_records())
    count = sum(language_counts.values())
    return {"record_count_scored": quality.records, "record_count_total": count,
            "characters": chars, "tokens": token_count, "language_counts": dict(language_counts),
            "language_estimate_counts":dict(language_estimate_counts),
            "language_confidence_mean_by_language":{name:confidence_totals[name]/n for name,n in language_estimate_counts.items()},
            "language_classification_methods":dict(language_method_counts),
            "quality": quality.as_dict() if hasattr(quality, "as_dict") else quality,
            "quality_sampling": "all records, streamed", "errors": errors,
            "processing_seconds": time.perf_counter() - started}
