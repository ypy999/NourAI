"""Phase 10.9 — Document leakage, translation-group alignment, and near-duplicate similarity audit."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import unicodedata


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_for_leakage(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _representative_shingle_hashes(text: str, *, k: int = 5, max_shingles: int = 8) -> list[str]:
    """Compute up to `max_shingles` deterministic min-hash word k-gram digests for texts >= 60 chars."""
    words = text.split()
    if len(words) < k or len(text) < 60:
        return []
    step = max(1, (len(words) - k + 1) // 32)
    digests = []
    for idx in range(0, len(words) - k + 1, step):
        gram = " ".join(words[idx : idx + k])
        digests.append(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest())
    return sorted(set(digests))[:max_shingles]


def audit_split_leakage_and_similarity(
    splits_dir: str | Path,
) -> dict[str, Any]:
    """Audit `train.jsonl`, `validation.jsonl`, and `test.jsonl` for exact, normalized, document, and translation-group leakage."""
    splits_path = Path(splits_dir).resolve()
    split_names = ("train", "validation", "test")
    split_files = {name: splits_path / f"{name}.jsonl" for name in split_names}
    for name, path in split_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Split file missing for {name}: {path}")

    exact_to_splits: dict[str, set[str]] = defaultdict(set)
    norm_to_splits: dict[str, set[str]] = defaultdict(set)
    doc_to_splits: dict[str, set[str]] = defaultdict(set)
    tgroup_to_splits: dict[str, set[str]] = defaultdict(set)
    tgroup_to_langs: dict[str, set[str]] = defaultdict(set)
    sgroup_to_splits: dict[str, set[str]] = defaultdict(set)

    train_shingle_to_docs: dict[str, set[str]] = defaultdict(set)
    eval_shingle_hits = 0
    eval_records_checked_for_shingles = 0

    split_record_counts: dict[str, int] = {name: 0 for name in split_names}
    split_doc_counts: dict[str, set[str]] = {name: set() for name in split_names}
    split_tgroup_counts: dict[str, set[str]] = {name: set() for name in split_names}

    # Pass 1: scan all splits
    for split_name in split_names:
        path = split_files[split_name]
        with path.open("rb") as stream:
            for raw_line in stream:
                row = json.loads(raw_line.decode("utf-8", errors="strict"))
                text = row["text"]
                source_id = row.get("source_id", "unknown")
                split_record_counts[split_name] += 1

                exact_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                exact_to_splits[exact_hash].add(split_name)

                norm_text = _normalize_for_leakage(text)
                norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
                norm_to_splits[norm_hash].add(split_name)

                doc_id = row.get("document_id") or row.get("original_document_id")
                qualified_doc = f"{source_id}:{doc_id}" if doc_id else f"{source_id}:rec:{row.get('record_id', exact_hash)}"
                if doc_id:
                    doc_to_splits[qualified_doc].add(split_name)
                    split_doc_counts[split_name].add(qualified_doc)

                tg_id = row.get("translation_group_id")
                if tg_id:
                    qualified_tg = f"{source_id}:{tg_id}"
                    tgroup_to_splits[qualified_tg].add(split_name)
                    tgroup_to_langs[qualified_tg].add(row.get("source_language_code") or row.get("original_language") or "unknown")
                    split_tgroup_counts[split_name].add(qualified_tg)

                sg_id = row.get("split_group_id")
                if sg_id:
                    sgroup_to_splits[str(sg_id)].add(split_name)

                shingles = _representative_shingle_hashes(norm_text)
                if shingles:
                    if split_name == "train":
                        for sh in shingles:
                            train_shingle_to_docs[sh].add(qualified_doc)
                    else:
                        eval_records_checked_for_shingles += 1
                        # Count if all min-hash shingles match a train document
                        matched_docs = set.intersection(*(train_shingle_to_docs.get(sh, set()) for sh in shingles))
                        if matched_docs:
                            eval_shingle_hits += 1

    exact_leakage = sum(1 for splits in exact_to_splits.values() if len(splits) > 1)
    normalized_leakage = sum(1 for splits in norm_to_splits.values() if len(splits) > 1)
    document_id_leakage = sum(1 for splits in doc_to_splits.values() if len(splits) > 1)
    translation_group_leakage = sum(1 for splits in tgroup_to_splits.values() if len(splits) > 1)
    split_group_id_leakage = sum(1 for splits in sgroup_to_splits.values() if len(splits) > 1)
    bilingual_pairs_intact = sum(1 for langs in tgroup_to_langs.values() if len(langs) == 2)

    passed = (
        exact_leakage == 0
        and normalized_leakage == 0
        and document_id_leakage == 0
        and translation_group_leakage == 0
        and split_group_id_leakage == 0
    )

    return {
        "phase": "10.9",
        "title": "Document Leakage & Similarity Audit",
        "status": "PASS" if passed else "FAIL",
        "created_at_utc": _utc_now(),
        "splits_directory": str(splits_path),
        "split_record_counts": split_record_counts,
        "unique_documents_with_explicit_id": len(doc_to_splits),
        "documents_by_split": {k: len(v) for k, v in split_doc_counts.items()},
        "unique_translation_groups": len(tgroup_to_splits),
        "bilingual_translation_groups_with_both_languages": bilingual_pairs_intact,
        "translation_groups_by_split": {k: len(v) for k, v in split_tgroup_counts.items()},
        "leakage_counts": {
            "exact_hash_leakage": exact_leakage,
            "normalized_hash_leakage": normalized_leakage,
            "document_id_leakage": document_id_leakage,
            "translation_group_leakage": translation_group_leakage,
            "split_group_id_leakage": split_group_id_leakage,
        },
        "similarity_audit": {
            "method": "5-word min-hash shingle signature (up to 8 shingles per record >= 60 chars)",
            "eval_records_checked": eval_records_checked_for_shingles,
            "full_signature_matches_in_train": eval_shingle_hits,
            "near_duplicate_rate": (
                eval_shingle_hits / eval_records_checked_for_shingles
                if eval_records_checked_for_shingles
                else 0.0
            ),
        },
        "acceptance_criteria": {
            "exact_leakage_zero": exact_leakage == 0,
            "normalized_leakage_zero": normalized_leakage == 0,
            "document_id_leakage_zero": document_id_leakage == 0,
            "translation_group_leakage_zero": translation_group_leakage == 0,
            "split_group_id_leakage_zero": split_group_id_leakage == 0,
        },
    }
