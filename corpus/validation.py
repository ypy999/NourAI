"""Integrity checks for imported records and prepared splits."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unicodedata
import re


def _normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def validate_jsonl(path, *, require_provenance=False, approved_source_ids=("tatoeba_cc0",)):
    result = {"path": str(path), "valid": True, "records": 0, "empty_records": 0,
              "duplicate_records": 0, "unknown_source_records": 0, "unknown_license_records": 0,
              "suspicious_records": 0, "errors": [], "warnings": [], "languages": {},
              "length_characters": {"min": None, "max": 0, "total": 0}}
    from data.language_analysis import classify_language
    path=Path(path)
    suspicious_pattern=re.compile(r"(.)\1{39,}|(?:lorem ipsum|sample text|placeholder text)",re.I)
    temp_parent=path.parent
    try:
        temp_ctx=tempfile.TemporaryDirectory(prefix="nourai-jsonl-validation-",dir=temp_parent)
    except OSError:
        temp_ctx=tempfile.TemporaryDirectory(prefix="nourai-jsonl-validation-")
    with temp_ctx as temp:
        db=sqlite3.connect(str(Path(temp)/"seen.sqlite3"))
        db.execute("CREATE TABLE seen(normalized_hash TEXT PRIMARY KEY, normalized_text TEXT NOT NULL)")
        try:
            with path.open("rb") as stream:
                for number, raw in enumerate(stream, 1):
                    try:
                        row = json.loads(raw.decode("utf-8", errors="strict"))
                        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                            raise ValueError("each line must be an object with string text")
                        text=row["text"]
                        if not text.strip():
                            result["empty_records"] += 1
                            raise ValueError("empty text record")
                        source_id=row.get("source_id")
                        license_id=row.get("license_id")
                        provenance_errors=[]
                        if source_id not in approved_source_ids:
                            result["unknown_source_records"] += 1
                            if require_provenance:
                                provenance_errors.append("source_id is unknown or not approved")
                        if license_id != "CC0-1.0":
                            result["unknown_license_records"] += 1
                            if require_provenance:
                                provenance_errors.append("record license is not an approved CC0-1.0 source")
                        if provenance_errors:
                            raise ValueError("; ".join(provenance_errors))
                        if "raw_text" in row:
                            if row["raw_text"] != text and not isinstance(row.get("normalized_text"),str):
                                raise ValueError("raw_text differs from text without normalized-stage trace")
                            raw_hash=hashlib.sha256(row["raw_text"].encode("utf-8",errors="strict")).hexdigest()
                            if row.get("raw_text_sha256") != raw_hash:
                                raise ValueError("raw_text_sha256 mismatch")
                        if "\ufffd" in text or any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
                            raise ValueError("replacement or surrogate character found")
                        normalized=_normalize(text)
                        digest=hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                        prior=db.execute("SELECT normalized_text FROM seen WHERE normalized_hash=?",(digest,)).fetchone()
                        if prior:
                            if prior[0] != normalized:
                                raise RuntimeError("SHA-256 collision in duplicate validation")
                            result["duplicate_records"] += 1
                            result["warnings"].append(f"line {number}: normalized duplicate")
                        else:
                            db.execute("INSERT INTO seen VALUES(?,?)",(digest,normalized))
                        language = classify_language(text)
                        result["languages"][language] = result["languages"].get(language, 0) + 1
                        length=len(text); result["length_characters"]["total"]+=length
                        result["length_characters"]["min"]=length if result["length_characters"]["min"] is None else min(result["length_characters"]["min"],length)
                        result["length_characters"]["max"]=max(result["length_characters"]["max"],length)
                        if suspicious_pattern.search(text):
                            result["suspicious_records"] += 1
                            result["warnings"].append(f"line {number}: suspicious repeated/placeholder text pattern")
                        result["records"] += 1
                        if number % 4096 == 0:
                            db.commit()
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        result["valid"] = False
                        if len(result["errors"]) < 100:
                            result["errors"].append(f"line {number}: {exc}")
        except OSError as exc:
            result["valid"] = False
            result["errors"].append(str(exc))
        finally:
            db.close()
    if result["duplicate_records"]:
        result["valid"] = False
        result["errors"].append(f"{result['duplicate_records']} normalized duplicate record(s) found")
    return result


def validate_splits(split_paths, *, temp_directory=None):
    """Check UTF-8/schema, nonempty splits, and normalized leakage disk-backed."""
    paths = {name: Path(path) for name, path in split_paths.items()}
    errors = []
    counts = {name: 0 for name in paths}
    if set(paths) != {"train", "validation", "test"}:
        return {"valid": False, "errors": ["expected train, validation, and test paths"], "split_counts": counts,
                "duplicate_leakage": None, "document_group_leakage": None}
    temp_parent = Path(temp_directory) if temp_directory else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nourai-phase8-validate-", dir=temp_parent) as temp:
        db = sqlite3.connect(str(Path(temp) / "seen.sqlite3"))
        db.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY, normalized TEXT NOT NULL, split TEXT NOT NULL)")
        db.execute("CREATE TABLE group_seen (group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        leakage = 0
        group_leakage = 0
        duplicates_within_split = 0
        try:
            for split, path in paths.items():
                try:
                    with path.open("rb") as stream:
                        for line_number, raw in enumerate(stream, 1):
                            try:
                                row = json.loads(raw.decode("utf-8", errors="strict"))
                                text = row["text"] if isinstance(row, dict) else None
                                if not isinstance(text, str) or not text.strip():
                                    raise ValueError("expected nonempty text")
                                normalized = _normalize(text)
                                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                                prior = db.execute("SELECT normalized, split FROM seen WHERE digest=?", (digest,)).fetchone()
                                if prior:
                                    if prior[0] != normalized:
                                        raise RuntimeError("SHA-256 collision in duplicate validation")
                                    if prior[1] != split:
                                        leakage += 1
                                    else:
                                        duplicates_within_split += 1
                                else:
                                    db.execute("INSERT INTO seen VALUES (?, ?, ?)", (digest, normalized, split))
                                group_id = row.get("split_group_id")
                                if isinstance(group_id, str) and group_id:
                                    prior_group = db.execute("SELECT split FROM group_seen WHERE group_id=?", (group_id,)).fetchone()
                                    if prior_group and prior_group[0] != split:
                                        group_leakage += 1
                                    elif not prior_group:
                                        db.execute("INSERT INTO group_seen VALUES (?,?)", (group_id, split))
                                counts[split] += 1
                            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
                                if len(errors) < 100:
                                    errors.append(f"{split}:{line_number}: {exc}")
                except OSError as exc:
                    errors.append(f"{split}: {exc}")
                db.commit()
        finally:
            db.close()
    if leakage:
        errors.append(f"{leakage} normalized duplicate records leaked across splits")
    if group_leakage:
        errors.append(f"{group_leakage} document group(s) leaked across splits")
    if duplicates_within_split:
        errors.append(f"{duplicates_within_split} duplicate records remain within a split")
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        errors.append("empty split(s): " + ", ".join(empty))
    return {"valid": not errors, "errors": errors, "split_counts": counts,
            "duplicate_leakage": leakage, "duplicates_within_splits": duplicates_within_split,
            "document_group_leakage": group_leakage}
