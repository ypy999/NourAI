"""Streaming text ingestion, conservative cleaning, de-duplication and splitting."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unicodedata

from configs.configuration import ProjectConfig, load_config
from data.deduplication import DiskDeduplicator
from data.provenance import canonical_record
from data.source_registry import SourceRegistry
from tokenizer.byte_tokenizer import ByteTokenizer


class DataPipelineError(RuntimeError):
    pass


@dataclass
class DataPipelineStats:
    files_seen: int = 0
    files_unreadable: int = 0
    unsupported_files: int = 0
    empty_files: int = 0
    records_seen: int = 0
    invalid_records: int = 0
    empty_records: int = 0
    duplicate_records: int = 0
    exact_duplicate_records: int = 0
    normalized_duplicate_records: int = 0
    oversized_records: int = 0
    short_records: int = 0
    malformed_unicode_records: int = 0
    removed_control_characters: int = 0
    accepted_records: int = 0
    train_records: int = 0
    validation_records: int = 0
    input_characters: int = 0
    output_characters: int = 0
    train_tokens: int = 0
    validation_tokens: int = 0
    errors: list = None
    errors_truncated: int = 0
    processing_seconds: float = 0.0
    source_estimate: dict = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def as_dict(self):
        return asdict(self)


def _record_error(stats, message):
    if len(stats.errors) < 100:
        stats.errors.append(message)
    else:
        stats.errors_truncated += 1


def estimate_sources(sources):
    """Estimate input bytes and file counts using metadata only, before reading."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    supported = []
    unsupported = []
    missing = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            for candidate in path.rglob("*"):
                if not candidate.is_file():
                    continue
                (supported if candidate.suffix.lower() in (".txt", ".jsonl") else unsupported).append(candidate)
        elif path.is_file() and path.suffix.lower() in (".txt", ".jsonl"):
            supported.append(path)
        elif path.exists():
            unsupported.append(path)
        else:
            missing.append(str(path))
    supported = sorted(set(path.resolve() for path in supported), key=lambda p: str(p).casefold())
    unsupported = sorted(set(path.resolve() for path in unsupported), key=lambda p: str(p).casefold())
    byte_count = 0
    for path in supported:
        try:
            byte_count += path.stat().st_size
        except OSError:
            pass
    return {"supported_files": len(supported), "unsupported_files": len(unsupported),
            "missing_sources": missing, "estimated_input_bytes": byte_count,
            "estimated_input_mib": round(byte_count / (1024 ** 2), 4)}


def clean_text(text, *, unicode_normalization="NFC", remove_control_chars=True,
               collapse_whitespace=True, preserve_paragraphs=False):
    """Conservatively clean one record without removing punctuation or diacritics.

    Unicode format characters (including Arabic ZWNJ and emoji ZWJ) are kept.
    Lone surrogate code points are removed because they cannot be UTF-8 encoded.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if unicode_normalization not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError("unsupported Unicode normalization form")
    text = unicodedata.normalize(unicode_normalization, text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned_lines = []
    blank_pending = False
    for line in text.split("\n"):
        kept = []
        for char in line:
            category = unicodedata.category(char)
            if category == "Cs" or (remove_control_chars and category == "Cc" and char != "\t"):
                continue
            kept.append(char)
        line = "".join(kept)
        line = " ".join(line.split()) if collapse_whitespace else line.strip()
        if not line:
            if preserve_paragraphs and cleaned_lines and not blank_pending:
                cleaned_lines.append("")
                blank_pending = True
            continue
        cleaned_lines.append(line)
        blank_pending = False
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


class DataPipeline:
    def __init__(self, config=None, tokenizer=None):
        self.config = config if isinstance(config, ProjectConfig) else load_config(config, tokenizer)
        self.tokenizer = tokenizer or ByteTokenizer()
        self.config.validate(self.tokenizer)
        self.source_registry = SourceRegistry()

    def _iter_records(self, paths, stats):
        files = []
        for source in paths:
            path = Path(source)
            if path.is_dir():
                for candidate in path.rglob("*"):
                    if not candidate.is_file():
                        continue
                    if candidate.suffix.lower() in (".txt", ".jsonl"):
                        files.append(candidate)
                    else:
                        stats.unsupported_files += 1
                        _record_error(stats, f"unsupported file type: {candidate}")
            elif path.is_file() and path.suffix.lower() in (".txt", ".jsonl"):
                files.append(path)
            elif path.exists():
                stats.unsupported_files += 1
                _record_error(stats, f"unsupported file type: {path}")
            else:
                _record_error(stats, f"unsupported or missing input: {path}")
        for path in sorted(set(files), key=lambda p: str(p).casefold()):
            stats.files_seen += 1
            valid_in_file = 0
            line_number = 0
            read_succeeded = True
            try:
                with path.open("rb") as stream:
                    max_line_bytes = self.config.max_document_chars * 4 + 3
                    while True:
                        raw = stream.readline(max_line_bytes)
                        if not raw:
                            break
                        line_number += 1
                        stats.records_seen += 1
                        if len(raw) >= max_line_bytes and not raw.endswith(b"\n"):
                            while raw and not raw.endswith(b"\n"):
                                raw = stream.readline(64 * 1024)
                            stats.oversized_records += 1
                            _record_error(stats, f"{path}:{line_number}: record exceeds safe input byte limit")
                            continue
                        try:
                            line = raw.decode("utf-8", errors="strict")
                            if path.suffix.lower() == ".jsonl":
                                obj = json.loads(line)
                                if not isinstance(obj, dict) or not isinstance(obj.get("text", obj.get("content")), str):
                                    raise ValueError("expected JSON object with string 'text' or 'content'")
                                text = obj.get("text", obj.get("content"))
                            else:
                                text = line
                        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                            stats.invalid_records += 1
                            _record_error(stats, f"{path}:{line_number}: {exc}")
                            continue
                        valid_in_file += 1
                        if len(text) > self.config.max_document_chars:
                            stats.oversized_records += 1
                            _record_error(stats, f"{path}:{line_number}: record exceeds max_document_chars")
                            continue
                        stats.input_characters += len(text)
                        malformed_count = sum(1 for char in text if unicodedata.category(char) == "Cs")
                        control_count = sum(1 for char in text if unicodedata.category(char) == "Cc" and char not in "\t\r\n")
                        if malformed_count:
                            stats.malformed_unicode_records += 1
                        stats.removed_control_characters += control_count if self.config.cleaning_remove_control_chars else 0
                        cleaned = clean_text(
                            text,
                            unicode_normalization=self.config.cleaning_unicode_normalization,
                            remove_control_chars=self.config.cleaning_remove_control_chars,
                            collapse_whitespace=self.config.cleaning_collapse_whitespace,
                            preserve_paragraphs=self.config.cleaning_preserve_paragraphs,
                        )
                        if not cleaned:
                            stats.empty_records += 1
                            continue
                        if len(cleaned) < self.config.cleaning_min_chars:
                            stats.short_records += 1
                            if self.config.cleaning_reject_short:
                                continue
                        output_record = canonical_record(obj if path.suffix.lower() == ".jsonl" else {"text": text},
                                                         cleaned_text=cleaned, source_file=path.resolve(),
                                                         source_line=line_number,
                                                         source_registry=self.source_registry)
                        # Keep legacy keys while preserving all canonical fields.
                        output_record["source"] = str(path.resolve())
                        output_record["line_number"] = line_number
                        yield output_record
            except OSError as exc:
                read_succeeded = False
                stats.files_unreadable += 1
                _record_error(stats, f"{path}: {exc}")
            if read_succeeded and valid_in_file == 0 and line_number > 0:
                stats.files_unreadable += 1
            elif read_succeeded and line_number == 0:
                stats.empty_files += 1

    def process(self, sources, output_dir):
        """Stream TXT/JSONL inputs to train.jsonl and validation.jsonl.

        Existing outputs are never overwritten. SQLite disk-backed exact dedup
        bounds Python memory use; source files remain untouched.
        """
        sources = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path, val_path = output_dir / "train.jsonl", output_dir / "validation.jsonl"
        if train_path.exists() or val_path.exists():
            raise FileExistsError("processed train.jsonl or validation.jsonl already exists; choose an empty output directory")
        stats = DataPipelineStats()
        stats.source_estimate = estimate_sources(sources)
        started = time.perf_counter()
        fd, db_name = tempfile.mkstemp(prefix="nourai-dedup-", suffix=".sqlite", dir=output_dir)
        os.close(fd)
        temp_outputs = []
        db = None
        try:
            with sqlite3.connect(db_name) as db:
                deduplicator = DiskDeduplicator(db)
                fd_train, tmp_train = tempfile.mkstemp(prefix="train-", suffix=".tmp", dir=output_dir)
                fd_val, tmp_val = tempfile.mkstemp(prefix="validation-", suffix=".tmp", dir=output_dir)
                temp_outputs = [tmp_train, tmp_val]
                os.close(fd_train); os.close(fd_val)
                with open(tmp_train, "w", encoding="utf-8", newline="\n") as train, open(tmp_val, "w", encoding="utf-8", newline="\n") as validation:
                    for record in self._iter_records(sources, stats):
                        text = record["text"]
                        duplicate_kind = deduplicator.add_if_unique(record)
                        if duplicate_kind is not None:
                            stats.duplicate_records += 1
                            if duplicate_kind == "exact":
                                stats.exact_duplicate_records += 1
                            else:
                                stats.normalized_duplicate_records += 1
                            continue
                        # Deterministic per-document hash split; dedup happens first.
                        digest = hashlib.sha256(f"{self.config.random_seed}:".encode() + text.encode("utf-8")).digest()
                        is_validation = int.from_bytes(digest[:8], "big") / 2**64 < self.config.validation_fraction
                        target = validation if is_validation else train
                        target.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stats.accepted_records += 1
                        stats.output_characters += len(text)
                        if is_validation:
                            stats.validation_records += 1
                            stats.validation_tokens += len(self.tokenizer.encode(text, add_eos=True))
                        else:
                            stats.train_records += 1
                            stats.train_tokens += len(self.tokenizer.encode(text, add_eos=True))
                        if stats.records_seen % self.config.io_batch_lines == 0:
                            db.commit()
                if stats.accepted_records == 0:
                    raise DataPipelineError("no valid, non-empty text records were found")
                if stats.train_records == 0 or stats.validation_records == 0:
                    raise DataPipelineError("split produced an empty training or validation set; add more unique records or adjust validation_fraction")
            os.replace(tmp_train, train_path)
            os.replace(tmp_val, val_path)
            temp_outputs.clear()
            stats.processing_seconds = time.perf_counter() - started
            return stats
        finally:
            if db is not None:
                db.close()
            for tmp in temp_outputs:
                try: os.unlink(tmp)
                except FileNotFoundError: pass
            try: os.unlink(db_name)
            except FileNotFoundError: pass
