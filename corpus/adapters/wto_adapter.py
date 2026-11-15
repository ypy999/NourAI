"""Streaming adapter for the WTO Public Corpus English/French parallel dataset."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterator
import zipfile

WTO_ADAPTER_VERSION = "wto_en_fr_adapter_v1"
DEFAULT_WTO_MEMBER = "Eng-Fra_UTF8/Eng-Fra-UTF8-00.txt"
EXPECTED_WTO_HEADER = [
    "ID",
    "IdxID",
    "SegID",
    "BTYear",
    "BTPath",
    "BTName",
    "SegSrc",
    "SegTgt",
    "Match",
    "IdxDomain",
]
ROW_START_PATTERN = re.compile(r"^\d+\t\d+\t\d+\t")


class WTOAdapterError(RuntimeError):
    """Raised when a WTO parallel corpus archive or TSV member violates integrity checks."""


def iter_reconstructed_wto_rows(
    stream,
    *,
    expected_header: list[str] | None = None,
) -> Iterator[tuple[int, list[str]]]:
    """Stream logical 10-column rows from a WTO UTF-8 TSV stream, reconstructing continuation lines."""
    expected = expected_header or EXPECTED_WTO_HEADER
    first_raw = stream.readline()
    if not first_raw:
        raise WTOAdapterError("WTO TSV member is empty; missing header row")
    header = first_raw.decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")
    if header != expected:
        raise WTOAdapterError(f"Unexpected WTO TSV header: {header!r} (expected {expected!r})")

    current_line_text: str | None = None
    current_start_line: int = 1
    for physical_line_number, raw_line in enumerate(stream, 2):
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if ROW_START_PATTERN.match(line):
            if current_line_text is not None:
                parts = current_line_text.split("\t")
                if len(parts) != 10:
                    raise WTOAdapterError(
                        f"Malformed WTO TSV row at line {current_start_line}: expected 10 columns, got {len(parts)}"
                    )
                yield current_start_line, parts
            current_line_text = line
            current_start_line = physical_line_number
        else:
            if current_line_text is None:
                raise WTOAdapterError(
                    f"Orphan continuation line before first valid WTO record at line {physical_line_number}"
                )
            current_line_text = f"{current_line_text}\n{line}"

    if current_line_text is not None:
        parts = current_line_text.split("\t")
        if len(parts) != 10:
            raise WTOAdapterError(
                f"Malformed WTO TSV row at line {current_start_line}: expected 10 columns, got {len(parts)}"
            )
        yield current_start_line, parts


def select_domain_stratified_documents(
    domain_to_docs: dict[str, set[str] | list[str]],
    *,
    seed: int | str = 1337,
    sample_fraction: float = 0.02,
    min_per_domain: int = 3,
    max_documents: int | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Deterministically select documents within each IdxDomain using SHA-256 ranking."""
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if min_per_domain < 1:
        raise ValueError("min_per_domain must be >= 1")

    selected_by_domain: dict[str, list[str]] = {}
    domain_summary: dict[str, dict[str, int]] = {}
    all_selected_ordered: list[tuple[str, str]] = []

    for domain in sorted(domain_to_docs):
        docs = sorted(set(domain_to_docs[domain]))
        available = len(docs)
        if available == 0:
            continue
        target_count = min(available, max(min_per_domain, math.ceil(available * sample_fraction)))
        ranked = sorted(
            docs,
            key=lambda doc_id: (
                hashlib.sha256(f"{seed}:{domain}:{doc_id}".encode("utf-8")).hexdigest(),
                doc_id,
            ),
        )
        chosen = sorted(ranked[:target_count])
        selected_by_domain[domain] = chosen
        domain_summary[domain] = {
            "available_documents": available,
            "selected_documents": len(chosen),
        }
        for doc_id in chosen:
            all_selected_ordered.append((domain, doc_id))

    if max_documents is not None and max_documents >= 0 and len(all_selected_ordered) > max_documents:
        capped_pairs = sorted(
            all_selected_ordered,
            key=lambda pair: (
                hashlib.sha256(f"{seed}:cap:{pair[0]}:{pair[1]}".encode("utf-8")).hexdigest(),
                pair[0],
                pair[1],
            ),
        )[:max_documents]
        capped_by_domain: dict[str, list[str]] = defaultdict(list)
        for dom, doc_id in capped_pairs:
            capped_by_domain[dom].append(doc_id)
        selected_by_domain = {dom: sorted(vals) for dom, vals in sorted(capped_by_domain.items())}
        for dom in domain_summary:
            domain_summary[dom]["selected_documents"] = len(selected_by_domain.get(dom, []))

    total_selected = sum(len(docs) for docs in selected_by_domain.values())
    summary = {
        "seed": seed,
        "sample_fraction": sample_fraction,
        "min_per_domain": min_per_domain,
        "max_documents": max_documents,
        "selected_documents": total_selected,
        "domain_document_sampling": domain_summary,
    }
    return selected_by_domain, summary


def scan_wto_member_domains(
    archive_path: str | Path,
    *,
    member_name: str = DEFAULT_WTO_MEMBER,
) -> dict[str, Any]:
    """Perform a single streaming pass over a WTO ZIP TSV member to inventory domains and rows."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise WTOAdapterError(f"WTO archive not found: {archive_path}")

    domain_docs: dict[str, set[str]] = defaultdict(set)
    paired_domain_docs: dict[str, set[str]] = defaultdict(set)
    total_rows = 0
    unpaired_rows = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if member_name not in zf.namelist():
                raise WTOAdapterError(f"WTO archive {archive_path} is missing member {member_name!r}")
            info = zf.getinfo(member_name)
            with zf.open(member_name, "r") as stream:
                for _, parts in iter_reconstructed_wto_rows(stream):
                    total_rows += 1
                    idx_id = parts[1].strip()
                    seg_src = parts[6].strip()
                    seg_tgt = parts[7].strip()
                    domain = parts[9].strip() or "UNKNOWN"
                    domain_docs[domain].add(idx_id)
                    if not seg_src or not seg_tgt:
                        unpaired_rows += 1
                    else:
                        paired_domain_docs[domain].add(idx_id)
    except zipfile.BadZipFile as exc:
        raise WTOAdapterError(f"Invalid WTO ZIP archive: {archive_path}") from exc

    return {
        "member_name": member_name,
        "member_uncompressed_bytes": info.file_size,
        "member_crc32": f"{info.CRC:08x}",
        "records_in_first_pass": total_rows,
        "unpaired_rows_skipped": unpaired_rows,
        "paired_rows_available": total_rows - unpaired_rows,
        "domain_to_docs": {dom: sorted(docs) for dom, docs in sorted(domain_docs.items())},
        "paired_domain_to_docs": {dom: sorted(docs) for dom, docs in sorted(paired_domain_docs.items())},
    }


def iter_wto_archive_records(
    archive_path: str | Path,
    *,
    member_name: str = DEFAULT_WTO_MEMBER,
    sampling_mode: str = "stratified_chunk00",
    seed: int | str = 1337,
    sample_fraction: float = 0.02,
    min_per_domain: int = 3,
    max_documents: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream paired English/French records from the WTO archive with full provenance."""
    archive_path = Path(archive_path)
    if sampling_mode not in {"stratified_chunk00", "full"}:
        raise WTOAdapterError(f"Unsupported WTO sampling_mode: {sampling_mode!r}")

    selected_docs: set[str] | None = None
    if sampling_mode == "stratified_chunk00":
        scan = scan_wto_member_domains(archive_path, member_name=member_name)
        selected_by_domain, _ = select_domain_stratified_documents(
            scan["domain_to_docs"],
            seed=seed,
            sample_fraction=sample_fraction,
            min_per_domain=min_per_domain,
            max_documents=max_documents,
        )
        selected_docs = {doc_id for docs in selected_by_domain.values() for doc_id in docs}

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            if member_name not in zf.namelist():
                raise WTOAdapterError(f"WTO archive {archive_path} is missing member {member_name!r}")
            with zf.open(member_name, "r") as stream:
                for line_number, parts in iter_reconstructed_wto_rows(stream):
                    wto_id = parts[0].strip()
                    idx_id = parts[1].strip()
                    seg_id = parts[2].strip()
                    bt_year = parts[3].strip()
                    bt_path = parts[4].strip()
                    bt_name = parts[5].strip()
                    seg_src = parts[6].strip()
                    seg_tgt = parts[7].strip()
                    match = parts[8].strip()
                    domain = parts[9].strip() or "UNKNOWN"

                    if not seg_src or not seg_tgt:
                        continue
                    if selected_docs is not None and idx_id not in selected_docs:
                        continue

                    translation_group_id = f"{idx_id}:{seg_id}"
                    source_file_ref = f"{archive_path}!{member_name}"
                    common_fields = {
                        "document_id": idx_id,
                        "original_document_id": idx_id,
                        "document_id_status": "SOURCE_PROVIDED",
                        "document_id_source_field": "IdxID",
                        "translation_group_id": translation_group_id,
                        "translation_group_method": "composite of official IdxID and SegID fields",
                        "segment_id": seg_id,
                        "wto_record_id": wto_id,
                        "document_path": bt_path,
                        "document_filename": bt_name,
                        "document_year": bt_year,
                        "domain": domain,
                        "alignment_match": match,
                        "source_file": source_file_ref,
                        "source_file_member": member_name,
                        "source_line": line_number,
                        "line_number": line_number,
                        "provenance_id_source": "WTO official ID/IdxID/SegID fields",
                        "adapter_version": WTO_ADAPTER_VERSION,
                    }

                    yield {
                        "text": seg_src,
                        "source_record_id": f"{wto_id}:eng",
                        "source_language_code": "eng",
                        "original_language": "eng",
                        "split_language": "english",
                        **common_fields,
                    }
                    yield {
                        "text": seg_tgt,
                        "source_record_id": f"{wto_id}:fra",
                        "source_language_code": "fra",
                        "original_language": "fra",
                        "split_language": "french",
                        **common_fields,
                    }
    except zipfile.BadZipFile as exc:
        raise WTOAdapterError(f"Invalid WTO ZIP archive: {archive_path}") from exc
