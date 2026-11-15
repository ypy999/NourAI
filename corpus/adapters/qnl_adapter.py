"""Streaming adapter for Qatar National Library Arabic OCR Corpus v2.

Reads `qnl_arabic_ocr_v2.zip` and `qnl_arabic_ocr_v2_metadata.csv` without
extracting the archive to disk. Verifies every text member against the embedded
`checksums.sha256` manifest, attaches the 9 official catalog metadata fields,
and computes the true Arabic Unicode script ratio (`arabic_chars / len(text)`).
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import zipfile

QNL_ADAPTER_VERSION = "qnl_arabic_ocr_v2_adapter_v1"

QNL_CSV_FIELD_MAP = (
    ("CALL #(ITEM)", "call_number"),
    ("Repository URL", "repository_url"),
    ("Catalog URL", "catalog_url"),
    ("AUTHOR", "author"),
    ("ADD AUTHOR", "additional_author"),
    ("PUB INFO", "publication_info"),
    ("TITLE", "title"),
    ("DESCRIPTION", "description"),
    ("VOLUME", "volume"),
)


def is_arabic_script_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def compute_arabic_script_ratio(text: str) -> float:
    """Return the proportion of characters in Unicode Arabic script blocks."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return 0.0
    arabic_count = sum(1 for ch in text if is_arabic_script_char(ch))
    return arabic_count / len(text)


def load_qnl_metadata_csv(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Parse `qnl_arabic_ocr_v2_metadata.csv` keyed by `RECORD #(ITEM)`."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"QNL metadata CSV not found: {path}")
    catalog: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "RECORD #(ITEM)" not in reader.fieldnames:
            raise ValueError(f"QNL metadata CSV is missing required 'RECORD #(ITEM)' column: {path}")
        for line_no, row in enumerate(reader, 2):
            record_id = (row.get("RECORD #(ITEM)") or "").strip()
            if not record_id:
                raise ValueError(f"{path}:{line_no}: empty 'RECORD #(ITEM)' value")
            if record_id in catalog:
                raise ValueError(f"{path}:{line_no}: duplicate QNL record ID {record_id!r}")
            catalog[record_id] = {
                target_key: (row.get(csv_col) or "").strip()
                for csv_col, target_key in QNL_CSV_FIELD_MAP
            }
    return catalog


def parse_checksums_manifest(raw_bytes: bytes) -> dict[str, str]:
    """Parse `checksums.sha256` lines (`<sha256> *<filename>` or `<sha256>  <filename>`)."""
    text = raw_bytes.decode("utf-8", errors="strict")
    checksums: dict[str, str] = {}
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"checksums.sha256:{line_no}: malformed checksum entry")
        digest = parts[0].lower()
        filename = parts[1].lstrip("*").strip()
        if not filename:
            raise ValueError(f"checksums.sha256:{line_no}: missing filename")
        checksums[filename] = digest
    return checksums


def resolve_metadata_csv_path(archive_path: str | Path, metadata_csv_path: str | Path | None = None) -> Path:
    if metadata_csv_path is not None:
        return Path(metadata_csv_path).resolve()
    archive_path = Path(archive_path).resolve()
    stem = archive_path.stem
    candidate = archive_path.with_name(f"{stem}_metadata.csv")
    if candidate.is_file():
        return candidate
    default_candidate = archive_path.parent / "qnl_arabic_ocr_v2_metadata.csv"
    if default_candidate.is_file():
        return default_candidate
    raise FileNotFoundError(
        f"QNL metadata CSV sidecar not found for {archive_path}; expected {candidate.name}"
    )


def extract_qnl_record_id(member_name: str) -> str:
    """Derive official QNL item record ID (e.g. `i10027154`) from `i10027154-00005595.txt`."""
    base = Path(member_name).stem
    record_id = base.split("-", 1)[0].strip()
    if not record_id:
        raise ValueError(f"cannot derive QNL record ID from member name {member_name!r}")
    return record_id


def select_qnl_documents(
    record_ids: list[str] | set[str],
    *,
    seed: int | str = 1337,
    sample_fraction: float = 0.02,
    min_documents: int = 3,
) -> set[str]:
    """Deterministically select QNL document IDs ranked by SHA-256(seed:QNL:record_id)."""
    docs = sorted(set(record_ids))
    if not docs:
        return set()
    target = min(len(docs), max(min_documents, math.ceil(len(docs) * sample_fraction)))
    ranked = sorted(
        docs,
        key=lambda rid: (hashlib.sha256(f"{seed}:QNL:{rid}".encode("utf-8")).hexdigest(), rid),
    )
    return set(ranked[:target])


def iter_qnl_archive_records(
    archive_path: str | Path,
    *,
    metadata_csv_path: str | Path | None = None,
    max_member_bytes: int = 2 * 1024**3,
    max_total_bytes: int = 8 * 1024**3,
    sampling_mode: str = "full",
    seed: int | str = 1337,
    sample_fraction: float = 0.02,
    min_documents: int = 3,
):
    """Stream validated QNL records with embedded checksums and official CSV metadata."""
    archive_path = Path(archive_path).resolve()
    if sampling_mode not in {"full", "deterministic_sample"}:
        yield {"error": f"{archive_path}: unsupported QNL sampling_mode {sampling_mode!r}"}
        return
    try:
        csv_path = resolve_metadata_csv_path(archive_path, metadata_csv_path)
        catalog = load_qnl_metadata_csv(csv_path)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
        yield {"error": f"{archive_path}: QNL metadata CSV error: {exc}"}
        return

    selected_ids: set[str] | None = None
    if sampling_mode == "deterministic_sample":
        selected_ids = select_qnl_documents(
            set(catalog.keys()),
            seed=seed,
            sample_fraction=sample_fraction,
            min_documents=min_documents,
        )

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = set(zf.namelist())
            if "checksums.sha256" not in names:
                yield {"error": f"{archive_path}: missing embedded checksums.sha256 manifest"}
                return
            checksums = parse_checksums_manifest(zf.read("checksums.sha256"))

            expanded_bytes = 0
            record_number = 0
            for info in zf.infolist():
                if info.is_dir() or info.filename == "checksums.sha256":
                    continue
                normalized = Path(info.filename.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    yield {"error": f"{archive_path}:{info.filename}: unsafe ZIP member path rejected"}
                    continue
                if not info.filename.casefold().endswith(".txt"):
                    yield {"error": f"{archive_path}:{info.filename}: unexpected non-TXT member in QNL archive"}
                    continue
                if info.file_size > max_member_bytes:
                    yield {"error": f"{archive_path}:{info.filename}: member exceeds size limit"}
                    continue
                expanded_bytes += info.file_size
                if expanded_bytes > max_total_bytes:
                    yield {"error": f"{archive_path}: expanded QNL archive exceeds total size limit"}
                    return

                expected_sha = checksums.get(info.filename) or checksums.get(normalized.name)
                if not expected_sha:
                    yield {"error": f"{archive_path}:{info.filename}: missing entry in checksums.sha256"}
                    continue

                try:
                    record_id = extract_qnl_record_id(normalized.name)
                except ValueError as exc:
                    yield {"error": f"{archive_path}:{info.filename}: {exc}"}
                    continue

                if record_id not in catalog:
                    yield {"error": f"{archive_path}:{info.filename}: record_id {record_id!r} missing from metadata CSV"}
                    continue

                if selected_ids is not None and record_id not in selected_ids:
                    continue

                with zf.open(info, "r") as member_stream:
                    raw_bytes = member_stream.read()

                actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                if actual_sha != expected_sha:
                    yield {
                        "error": (
                            f"{archive_path}:{info.filename}: SHA-256 mismatch "
                            f"(expected {expected_sha}, got {actual_sha})"
                        )
                    }
                    continue

                try:
                    text = raw_bytes.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    yield {"error": f"{archive_path}:{info.filename}: invalid UTF-8: {exc}"}
                    continue

                if not text.strip():
                    yield {"error": f"{archive_path}:{info.filename}: empty document text"}
                    continue

                record_number += 1
                ratio = compute_arabic_script_ratio(text)
                yield {
                    "text": text,
                    "raw_text": text,
                    "raw_text_sha256": actual_sha,
                    "content_hash": actual_sha,
                    "source": f"{archive_path}!{info.filename}",
                    "source_file": f"{archive_path}!{info.filename}",
                    "source_file_member": info.filename,
                    "record_number": record_number,
                    "source_line": None,
                    "source_record_id": record_id,
                    "document_id": record_id,
                    "original_document_id": record_id,
                    "language_code": "ara",
                    "source_language_code": "ara",
                    "original_language": "ara",
                    "detected_language": "arabic" if ratio >= 0.20 else "mixed",
                    "detected_language_method": "Unicode Arabic-script proportion >= 0.20 heuristic",
                    "split_language": "arabic",
                    "arabic_script_ratio": ratio,
                    "qnl_metadata": catalog[record_id],
                    "provenance_id_source": "QNL RECORD #(ITEM); original catalog record ID",
                    "document_id_status": "SOURCE_PROVIDED",
                    "adapter_version": QNL_ADAPTER_VERSION,
                }
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        yield {"error": f"{archive_path}: QNL archive error: {exc}"}
