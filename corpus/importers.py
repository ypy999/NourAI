"""Streaming import of text formats; source files are opened read-only."""
import bz2
import csv
import io
import json
from pathlib import Path
import tarfile
import zipfile


class ImportErrorRecord(ValueError):
    pass


def _text_value(record, text_fields=("text", "content", "sentence")):
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for field in text_fields:
            value = record.get(field)
            if isinstance(value, str):
                return value
    return None


_PROVENANCE_INPUT_FIELDS = (
    "document_id", "source_document_id", "doc_id", "article_id", "source_record_id",
    "source_record_number", "source_language_code", "language_code", "original_language",
    "sentence_domain", "domain", "domains", "source_name", "dataset_version",
)


def _provenance_fields(record):
    if not isinstance(record, dict):
        return {}
    result = {key: record[key] for key in _PROVENANCE_INPUT_FIELDS if key in record}
    if not result.get("document_id"):
        for alias in ("source_document_id", "doc_id", "article_id"):
            if record.get(alias) not in (None, ""):
                result["document_id"] = record[alias]
                break
    return result


def _iter_lines(stream, source, format_name, text_fields, max_record_bytes=8 * 1024 * 1024):
    number = 0
    while True:
        raw = stream.readline(max_record_bytes + 1)
        if not raw:
            break
        number += 1
        if len(raw) > max_record_bytes and not raw.endswith(b"\n"):
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(64 * 1024)
            yield {"error": f"{source}:{number}: record exceeds {max_record_bytes}-byte safety limit"}
            continue
        try:
            line = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        except UnicodeDecodeError as exc:
            yield {"error": f"{source}:{number}: invalid UTF-8: {exc}"}
            continue
        if not line.strip():
            continue
        try:
            if format_name == "jsonl":
                value = json.loads(line)
                text = _text_value(value, text_fields)
                if text is None:
                    raise ValueError("no configured string text field")
                yield {**_provenance_fields(value), "text": text, "source": str(source), "record_number": number}
            elif format_name == "tatoeba_cc0_tsv":
                fields = line.rstrip("\r\n").split("\t", 3)
                if len(fields) < 3 or not fields[0].isdigit() or not fields[1]:
                    raise ValueError("expected Tatoeba sentence_id<TAB>language<TAB>text row")
                yield {"sentence_id": fields[0], "language_code": fields[1], "text": fields[2],
                       "contributor_timestamp": fields[3] if len(fields) == 4 else None,
                       "source": str(source), "record_number": number, "license_id": "CC0-1.0"}
            else:
                yield {"text": line.rstrip("\r\n"), "source": str(source), "record_number": number}
        except (json.JSONDecodeError, ValueError) as exc:
            yield {"error": f"{source}:{number}: {exc}"}


def _iter_stream(stream, logical_name, text_fields, max_json_bytes):
    suffix = Path(logical_name).suffix.casefold()
    if suffix == ".csv" and "sentences_cc0" in logical_name.casefold():
        yield from _iter_lines(stream, logical_name, "tatoeba_cc0_tsv", text_fields)
        return
    kind = {".txt": "txt", ".text": "txt", ".jsonl": "jsonl", ".ndjson": "jsonl",
            ".csv": "csv", ".json": "json", ".parquet": "parquet"}.get(suffix)
    if logical_name.casefold().endswith(".tsv"):
        kind = "tatoeba_cc0_tsv"
    if kind in ("txt", "jsonl", "tatoeba_cc0_tsv"):
        yield from _iter_lines(stream, logical_name, kind, text_fields)
        return
    if kind == "csv":
        text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="strict", newline="")
        try:
            reader = csv.DictReader(text_stream)
            for number, row in enumerate(reader, 2):
                text = _text_value(row, text_fields)
                if text is None:
                    yield {"error": f"{logical_name}:{number}: no configured string text column"}
                else:
                    yield {**_provenance_fields(row), "text": text, "source": logical_name, "record_number": number}
        except (UnicodeDecodeError, csv.Error) as exc:
            yield {"error": f"{logical_name}: invalid CSV/UTF-8: {exc}"}
        return
    if kind == "json":
        # JSON documents are not line-streamable in the standard library; bound memory.
        payload = stream.read(max_json_bytes + 1)
        if len(payload) > max_json_bytes:
            yield {"error": f"{logical_name}: JSON exceeds bounded {max_json_bytes}-byte limit; convert to JSONL"}
            return
        try:
            value = json.loads(payload.decode("utf-8", errors="strict"))
            rows = value if isinstance(value, list) else [value]
            for number, row in enumerate(rows, 1):
                text = _text_value(row, text_fields)
                if text is None:
                    yield {"error": f"{logical_name}:{number}: no configured string text field"}
                else:
                    yield {**_provenance_fields(row), "text": text, "source": logical_name, "record_number": number}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            yield {"error": f"{logical_name}: invalid UTF-8 JSON: {exc}"}
        return
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            yield {"error": "Parquet support requires optional pyarrow; no dependency was installed"}
            return
        try:
            table_file = pq.ParquetFile(stream)
            number = 0
            for batch in table_file.iter_batches(batch_size=4096):
                columns = set(batch.schema.names)
                column = next((name for name in text_fields if name in columns), None)
                if column is None:
                    yield {"error": f"{logical_name}: no configured text column"}
                    return
                for text in batch.column(column).to_pylist():
                    number += 1
                    if isinstance(text, str):
                        yield {"text": text, "source": logical_name, "record_number": number}
                    else:
                        yield {"error": f"{logical_name}:{number}: text value is not a string"}
        except Exception as exc:
            yield {"error": f"{logical_name}: Parquet read failed: {exc}"}
        return
    yield {"error": f"{logical_name}: unsupported format"}


def iter_source_records(path, *, text_fields=("text", "content", "sentence"),
                        max_json_bytes=64 * 1024 * 1024, max_zip_member_bytes=2 * 1024**3,
                        max_archive_total_bytes=4 * 1024**3):
    """Yield record/error mappings. ZIP entries are streamed, never extracted."""
    path = Path(path)
    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".bz2"):
        try:
            if tarfile.is_tarfile(path):
                with tarfile.open(path, mode="r:bz2") as archive:
                    expanded = 0
                    for member in archive:
                        normalized = Path(member.name.replace("\\", "/"))
                        if normalized.is_absolute() or ".." in normalized.parts:
                            yield {"error": f"{path}:{member.name}: unsafe TAR path rejected"}
                            continue
                        if not member.isfile():
                            continue
                        if member.size > max_zip_member_bytes:
                            yield {"error": f"{path}:{member.name}: TAR member exceeds configured size limit"}
                            continue
                        expanded += member.size
                        if expanded > max_archive_total_bytes:
                            yield {"error": f"{path}: expanded TAR content exceeds configured total size limit"}
                            return
                        if member.name.casefold().endswith((".bz2", ".gz", ".xz", ".zip", ".tar")):
                            yield {"error": f"{path}:{member.name}: nested compressed archive is not imported"}
                            continue
                        stream = archive.extractfile(member)
                        if stream is None:
                            yield {"error": f"{path}:{member.name}: could not open TAR member"}
                            continue
                        with stream:
                            yield from _iter_stream(stream, member.name, text_fields, max_json_bytes)
                return
            with bz2.open(path, "rb") as stream:
                name = path.name[:-4]
                yield from _iter_stream(stream, name, text_fields, max_json_bytes)
        except (OSError, EOFError, tarfile.TarError, ValueError) as exc:
            yield {"error": f"{path}: compressed source is corrupt or unreadable: {exc}"}
        return
    if path.suffix.casefold() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                expanded = 0
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    normalized = Path(info.filename.replace("\\", "/"))
                    if normalized.is_absolute() or ".." in normalized.parts:
                        yield {"error": f"{path}:{info.filename}: unsafe ZIP path rejected"}
                        continue
                    if info.file_size > max_zip_member_bytes:
                        yield {"error": f"{path}:{info.filename}: ZIP member exceeds configured size limit"}
                        continue
                    expanded += info.file_size
                    if expanded > max_archive_total_bytes:
                        yield {"error": f"{path}: expanded ZIP content exceeds configured total size limit"}
                        return
                    if info.filename.casefold().endswith((".bz2", ".gz", ".xz", ".zip")):
                        yield {"error": f"{path}:{info.filename}: nested compressed archive is not imported"}
                        continue
                    with archive.open(info, "r") as stream:
                        yield from _iter_stream(stream, info.filename, text_fields, max_json_bytes)
        except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
            yield {"error": f"{path}: ZIP source is corrupt or unreadable: {exc}"}
        return
    try:
        with path.open("rb") as stream:
            yield from _iter_stream(stream, path.name, text_fields, max_json_bytes)
    except OSError as exc:
        yield {"error": f"{path}: source file is unreadable: {exc}"}
