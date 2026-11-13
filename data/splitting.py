"""Reproducible three-way, disk-backed dataset splitting."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from .deduplication import DiskDeduplicator, normalize_duplicate_key
from .language_analysis import classify_language


SPLITS = ("train", "validation", "test")


@dataclass
class DatasetSplitStats:
    input_records: int = 0
    empty_records: int = 0
    duplicate_records: int = 0
    exact_duplicate_records: int = 0
    normalized_duplicate_records: int = 0
    unique_records: int = 0
    split_counts: dict = field(default_factory=lambda: {name: 0 for name in SPLITS})
    language_counts: dict = field(default_factory=dict)
    requested_ratios: dict = field(default_factory=dict)
    actual_ratios: dict = field(default_factory=dict)
    seed: int = 1337
    warnings: list = field(default_factory=list)
    processing_seconds: float = 0.0
    source_import_stats: dict = field(default_factory=dict)
    source_estimate: dict = field(default_factory=dict)
    group_count: int = 0
    records_without_document_id: int = 0
    group_split_counts: dict = field(default_factory=dict)
    grouping_policy: str = "document_id when present; otherwise each stable record is its own group"

    def as_dict(self):
        return asdict(self)


def _validate_ratios(ratios):
    if not isinstance(ratios, dict) or set(ratios) != set(SPLITS):
        raise ValueError("ratios must contain exactly train, validation, and test")
    values = []
    for name in SPLITS:
        value = ratios[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} ratio must be a non-negative number")
        values.append(float(value))
    if abs(sum(values) - 1.0) > 1e-9 or values[0] <= 0:
        raise ValueError("split ratios must sum to 1 and train ratio must be positive")
    return dict(zip(SPLITS, values))


def _allocate(count, ratios):
    exact = [count * ratios[name] for name in SPLITS]
    quotas = [int(value) for value in exact]
    left = count - sum(quotas)
    order = sorted(range(3), key=lambda index: (-(exact[index] - quotas[index]), index))
    for index in order[:left]:
        quotas[index] += 1
    return dict(zip(SPLITS, quotas))


def _make_split_group(record, group_field, normalized_hash):
    """Namespace source-local document IDs; never collapse missing IDs together."""
    group_value = record.get(group_field) if group_field else None
    if group_value not in (None, ""):
        namespace = record.get("source_id") or record.get("source") or ""
        method = "document_id"
    else:
        group_value = record.get("record_id") or normalized_hash
        namespace = record.get("source_id") or record.get("source") or ""
        method = "record_id" if record.get("record_id") else "independent_record_no_document_id"
    raw_key = f"{namespace}\0{group_field or 'record'}\0{group_value}"
    group_key = hashlib.sha256(raw_key.encode("utf-8", errors="strict")).hexdigest()
    return group_key, method


def _assign_group_splits(db, ratios, seed, stratify):
    grouped = {}
    for group_key, language, priority, count in db.execute(
            "SELECT group_key,language,MIN(priority),COUNT(*) FROM documents GROUP BY group_key,language"):
        row = grouped.setdefault(group_key, {"languages": {}, "priority": priority, "count": 0})
        row["languages"][language] = count
        row["count"] += count
        row["priority"] = min(row["priority"], priority)
    categories = {}
    for group_key, row in grouped.items():
        dominant = sorted(row["languages"].items(), key=lambda item: (-item[1], item[0]))[0][0]
        category = dominant if stratify else "all"
        categories.setdefault(category, []).append((row["priority"], group_key, row["count"]))
    assignments = []
    group_counts = {name: {split: 0 for split in SPLITS} for name in SPLITS}
    for category, groups in categories.items():
        total_records = sum(count for _, _, count in groups)
        target = {split: total_records * ratios[split] for split in SPLITS}
        current = {split: 0 for split in SPLITS}
        positive_splits = [split for split in SPLITS if ratios[split] > 0]
        assigned_keys = set()
        if len(groups) >= len(positive_splits):
            # Guarantee requested splits receive a whole group when possible.
            # Reserve smaller groups for underweighted holdouts to minimize
            # record-count distortion; this does not split a group.
            size_order = sorted(groups, key=lambda item: (item[2], item[0], item[1]))
            reserved = {}
            holdouts = [split for split in positive_splits if split != "train"]
            for split, group in zip(holdouts, size_order):
                reserved[group[1]] = split
            remaining_for_train = [group for group in reversed(size_order) if group[1] not in reserved]
            if "train" in positive_splits and remaining_for_train:
                reserved[remaining_for_train[0][1]] = "train"
            by_key = {group_key: (priority, group_key, count) for priority, group_key, count in groups}
            for group_key, split in reserved.items():
                current[split] += by_key[group_key][2]
                assignments.append((group_key, split, category))
                assigned_keys.add(group_key)
        # Hash priority makes assignment deterministic and input-order independent.
        for _, group_key, count in sorted(groups):
            if group_key in assigned_keys:
                continue
            def score(candidate):
                return sum(((current[split] + (count if split == candidate else 0) - target[split]) ** 2)
                           / max(target[split], 1.0) for split in SPLITS)
            chosen = min(SPLITS, key=lambda split: (score(split), SPLITS.index(split)))
            current[chosen] += count
            assignments.append((group_key, chosen, category))
            group_counts[chosen][category] = group_counts[chosen].get(category, 0) + 1
    db.executemany("INSERT OR REPLACE INTO group_assignments(group_key,split,language) VALUES(?,?,?)", assignments)
    return len(grouped), group_counts


def split_records(records, output_dir, *, ratios=None, seed=1337, stratify=True, group_field="document_id",
                  checkpoint_path=None, resume=False, source_fingerprint=None,
                  checkpoint_batch_size=256):
    """Write deduplicated JSONL splits without retaining corpus rows in RAM.

    Duplicate filtering occurs globally before assignment, so an exact or
    normalized duplicate cannot leak across any split. Stable hashes plus
    largest-remainder quotas make a split reproducible for the same input,
    order-independent, seed, and ratio configuration.
    """
    ratios = _validate_ratios({"train": 0.90, "validation": 0.05, "test": 0.05} if ratios is None else ratios)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(stratify, bool):
        raise ValueError("stratify must be boolean")
    if group_field is not None and (not isinstance(group_field, str) or not group_field.strip()):
        raise ValueError("group_field must be a non-empty string or None")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    if isinstance(checkpoint_batch_size, bool) or not isinstance(checkpoint_batch_size, int) or checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be a positive integer")
    if checkpoint_path is None and resume:
        raise ValueError("resume requires checkpoint_path")
    if checkpoint_path is not None and (not isinstance(source_fingerprint, str) or not source_fingerprint.strip()):
        raise ValueError("checkpointing requires a stable source_fingerprint")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {name: output_dir / f"{name}.jsonl" for name in SPLITS}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("one or more split outputs already exist; choose a new output directory")
    stats = DatasetSplitStats(seed=seed, requested_ratios=ratios)
    started = time.perf_counter()
    persistent_checkpoint = checkpoint_path is not None
    if persistent_checkpoint:
        db_path = Path(checkpoint_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not db_path.is_file():
            raise FileNotFoundError(f"split checkpoint not found: {db_path}")
        if not resume and db_path.exists():
            raise FileExistsError(f"split checkpoint already exists: {db_path}")
        db_name = str(db_path)
    else:
        fd, db_name = tempfile.mkstemp(prefix="nourai-split-", suffix=".sqlite", dir=output_dir)
        os.close(fd)
    temp_paths = {}
    db = None
    try:
        with sqlite3.connect(db_name) as db:
            deduplicator = DiskDeduplicator(db)
            db.execute("CREATE TABLE IF NOT EXISTS documents (normalized_hash TEXT PRIMARY KEY, language TEXT NOT NULL, priority TEXT NOT NULL, payload TEXT NOT NULL, group_key TEXT NOT NULL)")
            document_columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if "group_key" not in document_columns:
                if resume:
                    raise ValueError("checkpoint predates group-aware splitting and cannot be resumed safely")
                db.execute("ALTER TABLE documents ADD COLUMN group_key TEXT NOT NULL DEFAULT ''")
            db.execute("CREATE TABLE IF NOT EXISTS group_assignments (group_key TEXT PRIMARY KEY, split TEXT NOT NULL, language TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS processing_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS processing_state (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
            signature = json.dumps({"seed": seed, "ratios": ratios, "stratify": stratify, "group_field": group_field,
                                    "source_fingerprint": source_fingerprint}, sort_keys=True)
            existing_signature = db.execute("SELECT value FROM processing_meta WHERE key='signature'").fetchone()
            state_keys = ("cursor", "input_records", "empty_records", "duplicate_records",
                          "exact_duplicate_records", "normalized_duplicate_records", "unique_records",
                          "records_without_document_id")
            if resume:
                if existing_signature is None or existing_signature[0] != signature:
                    raise ValueError("checkpoint settings or source fingerprint do not match")
                saved_state = dict(db.execute("SELECT key, value FROM processing_state"))
                state = {key: int(saved_state.get(key, 0)) for key in state_keys}
            else:
                db.execute("INSERT INTO processing_meta(key, value) VALUES ('signature', ?)", (signature,))
                state = {key: 0 for key in state_keys}
                db.executemany("INSERT INTO processing_state(key, value) VALUES (?, ?)", state.items())
                db.commit()
            cursor = state["cursor"]
            input_index = 0
            for item in records:
                if input_index < cursor:
                    input_index += 1
                    continue
                state["input_records"] += 1
                if isinstance(item, str):
                    record = {"text": item}
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    record = dict(item)
                else:
                    raise ValueError("records must be strings or mappings containing string 'text'")
                if not record["text"].strip():
                    state["empty_records"] += 1
                else:
                    record.setdefault("source", "")
                    record.setdefault("line_number", 0)
                    if group_field is None or record.get(group_field) in (None, ""):
                        state["records_without_document_id"] += 1
                    duplicate_kind = deduplicator.add_if_unique(record)
                    if duplicate_kind:
                        state["duplicate_records"] += 1
                        state["exact_duplicate_records" if duplicate_kind == "exact" else "normalized_duplicate_records"] += 1
                    else:
                        # Large, provenance-rich corpora may already carry a
                        # validated split stratum. Reuse it when recognized;
                        # otherwise retain the historical classifier behavior.
                        language = record.get("split_language")
                        if language not in ("arabic", "english", "french", "mixed", "unclassified"):
                            language = classify_language(record["text"])
                        grouping = language if stratify else "all"
                        normalized_hash = hashlib.sha256(normalize_duplicate_key(record["text"]).encode("utf-8")).hexdigest()
                        group_key, group_method = _make_split_group(record, group_field, normalized_hash)
                        record["split_group_id"] = group_key
                        record["split_group_method"] = group_method
                        priority = hashlib.sha256(f"{seed}:{group_key}:{grouping}".encode("utf-8")).hexdigest()
                        db.execute("INSERT INTO documents(normalized_hash,language,priority,payload,group_key) VALUES(?,?,?,?,?)",
                                   (normalized_hash, grouping, priority, json.dumps(record, ensure_ascii=False), group_key))
                        state["unique_records"] += 1
                input_index += 1
                state["cursor"] = input_index
                if state["cursor"] % checkpoint_batch_size == 0:
                    db.executemany("UPDATE processing_state SET value=? WHERE key=?",
                                   [(value, key) for key, value in state.items()])
                    db.commit()
            if input_index < cursor:
                raise ValueError("resumed input ended before the saved checkpoint cursor")
            db.executemany("UPDATE processing_state SET value=? WHERE key=?",
                           [(value, key) for key, value in state.items()])
            db.commit()
            stats.input_records = state["input_records"]
            stats.empty_records = state["empty_records"]
            stats.duplicate_records = state["duplicate_records"]
            stats.exact_duplicate_records = state["exact_duplicate_records"]
            stats.normalized_duplicate_records = state["normalized_duplicate_records"]
            stats.unique_records = state["unique_records"]
            stats.records_without_document_id = state["records_without_document_id"]
            db.execute("DELETE FROM group_assignments")
            group_count, group_split_counts = _assign_group_splits(db, ratios, seed, stratify)
            stats.group_count = group_count
            stats.group_split_counts = group_split_counts
            stats.records_without_document_id = db.execute("SELECT COUNT(*) FROM documents WHERE json_extract(payload, ?) IS NULL", ("$." + group_field if group_field else "$.document_id",)).fetchone()[0]
            category_sizes = db.execute("SELECT language,COUNT(*) FROM documents GROUP BY language").fetchall()
            for language, count in category_sizes:
                if count < 3 and any(ratios[name] > 0 for name in ("validation", "test")):
                    stats.warnings.append(f"{language}: {count} unique record(s); group sizes may leave a requested split empty")
            counters = {language: {name: 0 for name in SPLITS} for language, _ in category_sizes}
            file_handles = {}
            try:
                for name, destination in destinations.items():
                    fd_out, path = tempfile.mkstemp(prefix=f"{name}-", suffix=".tmp", dir=output_dir)
                    os.close(fd_out)
                    temp_paths[name] = Path(path)
                    file_handles[name] = open(path, "w", encoding="utf-8", newline="\n")
                query = ("SELECT documents.language,documents.priority,documents.payload,group_assignments.split "
                         "FROM documents JOIN group_assignments USING(group_key) "
                         "ORDER BY documents.language,documents.priority,documents.normalized_hash")
                for language, priority, payload, split in db.execute(query):
                    file_handles[split].write(payload + "\n")
                    counters[language][split] += 1
                    stats.split_counts[split] += 1
                    stats.language_counts.setdefault(split, {})
                    stats.language_counts[split][language] = stats.language_counts[split].get(language, 0) + 1
            finally:
                for handle in file_handles.values():
                    handle.close()
        for name in SPLITS:
            temp_paths[name].replace(destinations[name])
            del temp_paths[name]
        if stats.unique_records == 0:
            stats.warnings.append("No unique records were available for splitting")
        stats.actual_ratios = {name: stats.split_counts[name] / stats.unique_records if stats.unique_records else 0.0 for name in SPLITS}
        stats.processing_seconds = time.perf_counter() - started
        return stats
    finally:
        if db is not None:
            db.close()
        for path in temp_paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if not persistent_checkpoint:
            try:
                Path(db_name).unlink()
            except FileNotFoundError:
                pass


def split_sources(sources, output_dir, *, config=None, tokenizer=None, ratios=None, seed=None, stratify=True,
                  group_field="document_id",
                  checkpoint_path=None, resume=False, checkpoint_batch_size=256):
    """Stream clean TXT/JSONL records from sources and create three splits.

    Sources must be original inputs, not the legacy phase-4 train/validation
    files, to avoid re-splitting an already partitioned dataset.
    """
    from .processor import DataPipeline, DataPipelineStats, estimate_sources
    sources = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
    source_estimate = estimate_sources(sources)
    pipeline = DataPipeline(config=config, tokenizer=tokenizer)
    import_stats = DataPipelineStats()

    def counted_records():
        for record in pipeline._iter_records(sources, import_stats):
            import_stats.accepted_records += 1
            import_stats.output_characters += len(record["text"])
            yield record

    actual_seed = pipeline.config.random_seed if seed is None else seed
    fingerprint = _fingerprint_sources(sources, pipeline.config) if checkpoint_path is not None else None
    result = split_records(counted_records(), output_dir, ratios=ratios, seed=actual_seed, stratify=stratify,
                           group_field=group_field,
                           checkpoint_path=checkpoint_path, resume=resume,
                           source_fingerprint=fingerprint, checkpoint_batch_size=checkpoint_batch_size)
    result.source_import_stats = import_stats.as_dict()
    result.source_estimate = source_estimate
    return result


def _fingerprint_sources(sources, config):
    """Stream source bytes into a digest; used only when resumability is enabled."""
    digest = hashlib.sha256()
    paths = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in (".txt", ".jsonl"))
        elif path.is_file() and path.suffix.lower() in (".txt", ".jsonl"):
            paths.append(path)
        else:
            digest.update(f"unsupported:{path.resolve()}".encode("utf-8"))
    for path in sorted(set(paths), key=lambda item: str(item).casefold()):
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as source_file:
            while True:
                chunk = source_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    digest.update(json.dumps({"cleaning_unicode_normalization": config.cleaning_unicode_normalization,
                              "cleaning_remove_control_chars": config.cleaning_remove_control_chars,
                              "cleaning_collapse_whitespace": config.cleaning_collapse_whitespace,
                              "cleaning_preserve_paragraphs": config.cleaning_preserve_paragraphs,
                              "cleaning_min_chars": config.cleaning_min_chars,
                              "cleaning_reject_short": config.cleaning_reject_short,
                              "max_document_chars": config.max_document_chars}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
