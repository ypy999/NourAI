"""Reproducibility audit and planning for the frozen Phase 10.2 corpus.

This module does not regenerate corpus payloads. It records, with hashes and
schema evidence, what each existing artifact is, whether the code in this
repository can produce it, and what blocks a clean re-run. The companion
``plan`` command validates inputs, sidecars and split decisions before any
expensive work is approved.

Design rules:
- missing source metadata is never defaulted;
- the frozen ``phase10_2`` tree is never used as an output directory;
- every verdict is derived from on-disk evidence or an explicit blocker.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .acquisition import AcquisitionError, sha256_file, verify_download_provenance
from .licensing import get_source

PHASE10_2_PROCESSED = Path("data/processed/phase10_2")
PHASE10_2_RAW = Path("data/raw/external/phase10_2")
PHASE10_2_REPORTS = Path("reports/phase10/phase10_2")

# Registry format -> (support level, explanation). Only formats whose records
# can be represented by corpus.importers without losing required provenance
# are marked SUPPORTED.
FORMAT_SUPPORT = {
    "tatoeba_cc0_tsv_bz2": (
        "SUPPORTED",
        "corpus.importers reads the CC0 TSV tar.bz2 and keeps sentence_id and language.",
    ),
    "zip_txt_with_official_metadata_csv": (
        "PARTIAL",
        "Generic ZIP/TXT import reads text only; QNL document_id, official record "
        "number and metadata CSV fields are not derived by corpus.importers.",
    ),
    "zip_tab_delimited_parallel_corpus": (
        "UNSUPPORTED",
        "Generic TXT import would treat each tab-separated parallel line as one "
        "record and lose document/segment alignment; no WTO adapter exists.",
    ),
}

KIND_BY_NAME = (
    (".partial", "incomplete_attempt"),
    ("imported.jsonl", "imported"),
    ("normalized.jsonl", "normalized"),
    ("train.jsonl", "split"),
    ("validation.jsonl", "split"),
    ("test.jsonl", "split"),
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace(os.sep, "/")


def classify_kind(relative_path):
    name = Path(relative_path).name
    name_cf = name.casefold()
    suffix = Path(relative_path).suffix.casefold()
    if suffix in {".partial", ".part"}:
        return "incomplete_attempt"
    if name_cf.endswith((".provenance.json", ".import.json", ".normalized.json")) or name_cf == "run_manifest.json":
        return "provenance_sidecar"
    if suffix == ".bin":
        return "download_fragment"
    for needle, kind in KIND_BY_NAME:
        if name == needle or relative_path.endswith("/" + needle):
            return kind
    if suffix == ".jsonl":
        parts = {part.casefold() for part in Path(relative_path).parts}
        if name_cf.endswith((".normalized.jsonl", ".normalized.cleaned.jsonl")) or "normalized" in parts:
            return "normalized"
        if name_cf.endswith(".imported.jsonl") or "imported" in parts:
            return "imported"
        if "splits" in parts:
            return "split"
    if suffix in {".zip", ".csv"} or name_cf.endswith((".tar.bz2", ".tsv.bz2")):
        return "raw_source"
    if relative_path.startswith("reports/"):
        return "report"
    return "other"


def first_record_keys(path, *, limit=1):
    """Return schema keys of the first JSONL record(s) without loading the file."""
    keys = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="strict") as stream:
            for _ in range(limit):
                line = stream.readline()
                if not line.strip():
                    break
                value = json.loads(line)
                if isinstance(value, dict):
                    keys.append(list(value.keys()))
                else:
                    keys.append([f"<{type(value).__name__}>"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return keys


def sidecar_path(path):
    path = Path(path)
    for suffix in (".provenance.json", ".import.json", ".normalized.json"):
        candidate = path.with_name(path.name + suffix)
        if candidate.is_file():
            return candidate
    return None


def _read_json_safe(path):
    if path is None or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _producer_verdict(root, relative_path, kind, keys, *, full_path=None):
    """Explain how a file was produced and whether this repository can rebuild it."""
    if kind == "raw_source":
        return ("YES", "raw artifact is preserved; content is pinned by its SHA-256")
    if kind == "provenance_sidecar":
        return ("N/A", "download provenance sidecar; evidence for the raw artifact, not a payload")
    if kind == "incomplete_attempt":
        return (
            "NO",
            "aborted or interrupted run residue; reports/phase10/phase10_2/wto_csv_parser_attempt.json "
            "records ABORTED_NOT_PUBLISHED and no generator exists in the repository",
        )
    if kind == "download_fragment":
        return (
            "NO",
            "ranged-download fragment, not a complete archive; it is not an importer input and "
            "cannot be regenerated byte-for-byte",
        )
    if kind == "imported":
        import_meta = _read_json_safe(Path(full_path).with_name(Path(full_path).name + ".import.json")) if full_path else None
        adhoc = {"qnl_metadata", "arabic_script_ratio", "provenance_id_source", "document_id_status"}
        present = adhoc & set(keys or [])
        if present and not (import_meta and import_meta.get("adapter_version")):
            return (
                "NO",
                "contains ad-hoc enrichment fields not written by corpus/pipeline.py: "
                + ", ".join(sorted(present)),
            )
        if import_meta and import_meta.get("source_sha256") and import_meta.get("output_sha256"):
            return ("YES", "imported with verified .import.json sidecar and pinned source_sha256")
        return ("PARTIAL", "shape resembles corpus.pipeline.import_source_file but no run sidecar exists")
    if kind == "normalized":
        norm_meta = _read_json_safe(Path(full_path).with_name(Path(full_path).name + ".normalized.json")) if full_path else None
        missing = {"raw_text", "normalized_text", "cleaned_text"} - set(keys or [])
        stage_hashes = {"raw_text_sha256", "normalized_text_sha256", "cleaned_text_sha256"} <= set(keys or [])
        if missing:
            if norm_meta and norm_meta.get("retain_stage_texts") is False and norm_meta.get("input_sha256") and stage_hashes:
                return (
                    "YES",
                    "compact traceable normalization with .normalized.json sidecar, input_sha256, and stage hashes",
                )
            return (
                "NO",
                "data.normalization.transform_record and canonical_record keep "
                + ", ".join(sorted(missing))
                + "; this file dropped them, so the recorded transformation cannot be replayed",
            )
        if norm_meta and norm_meta.get("input_sha256") and norm_meta.get("output_sha256"):
            return ("YES", "normalized with verified .normalized.json sidecar and pinned input_sha256")
        return ("PARTIAL", "schema matches transform_record but no import sidecar or run manifest exists")
    if kind == "split":
        if "split_group_id" in set(keys or []):
            if full_path is not None:
                manifest_candidates = [
                    Path(full_path).parent / "run_manifest.json",
                    Path(full_path).parent.parent / "run_manifest.json",
                ]
                for candidate in manifest_candidates:
                    manifest = _read_json_safe(candidate)
                    if manifest and manifest.get("status") == "COMPLETE":
                        return ("YES", f"reproducible split governed by verified {candidate.name}")
            return (
                "PARTIAL",
                "record schema matches data.splitting.split_sources, but the parent normalized "
                "inputs are not reproducible and split decisions used a non-default seed/ratio",
            )
        return ("NO", "split file lacks split_group_id and cannot be traced to a splitter run")
    return ("N/A", "report/evidence file; not a generated corpus payload")


def audit_artifacts(root, *, hash_files=True, include_reports=True, targets=None):
    """Inventory the frozen Phase 10.2 payload (or custom targets) and its evidence files."""
    root = Path(root).resolve()
    if targets is None:
        targets = [root / PHASE10_2_PROCESSED, root / PHASE10_2_RAW]
        if include_reports:
            targets.append(root / PHASE10_2_REPORTS)
    else:
        targets = [(root / Path(t)).resolve() if not Path(t).is_absolute() else Path(t).resolve() for t in targets]
    entries = []
    for target in targets:
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            relative = _relative(root, path)
            if any("pytest-" in part for part in Path(relative).parts):
                continue
            if path.name.endswith((".import.json", ".normalized.json")) or path.name == "run_manifest.json":
                kind = "provenance_sidecar"
            else:
                kind = classify_kind(relative)
            keys = first_record_keys(path) if kind in {"imported", "normalized", "split"} else None
            first_keys = keys[0] if keys else None
            verdict, reason = _producer_verdict(root, relative, kind, first_keys, full_path=path)
            sidecar = sidecar_path(path)
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if hash_files else None,
                    "kind": kind,
                    "schema_keys": first_keys,
                    "provenance_sidecar": _relative(root, sidecar) if sidecar else None,
                    "reproducible": verdict,
                    "reason": reason,
                }
            )
    summary = {
        "artifacts": len(entries),
        "bytes": sum(item["bytes"] for item in entries),
        "by_kind": {},
        "by_reproducibility": {},
    }
    for item in entries:
        summary["by_kind"][item["kind"]] = summary["by_kind"].get(item["kind"], 0) + 1
        summary["by_reproducibility"][item["reproducible"]] = (
            summary["by_reproducibility"].get(item["reproducible"], 0) + 1
        )
    return {
        "created_at_utc": _utc_now(),
        "root": str(root),
        "hash_files": hash_files,
        "summary": summary,
        "artifacts": entries,
    }


def format_support(source_id):
    """Return the importer support verdict for a registry source."""
    source = get_source(source_id)
    level, explanation = FORMAT_SUPPORT.get(
        source.get("format", ""),
        ("UNKNOWN", f"no importer support entry for format {source.get('format')!r}"),
    )
    return {"source_id": source_id, "format": source.get("format"), "support": level, "explanation": explanation}


def check_split_decisions(split_report, approved_config):
    """Compare a recorded split run against the approved project decisions."""
    findings = []
    requested = (split_report or {}).get("requested_ratios") or {}
    approved_ratios = approved_config.get("split_ratios") or {}
    for name in ("train", "validation", "test"):
        if requested.get(name) != approved_ratios.get(name):
            findings.append(
                {
                    "field": f"split_ratios.{name}",
                    "expected": approved_ratios.get(name),
                    "actual": requested.get(name),
                    "severity": "BLOCKER",
                }
            )
    if (split_report or {}).get("seed") != approved_config.get("random_seed"):
        findings.append(
            {
                "field": "random_seed",
                "expected": approved_config.get("random_seed"),
                "actual": (split_report or {}).get("seed"),
                "severity": "BLOCKER",
            }
        )
    grouping = (split_report or {}).get("grouping_field")
    if not grouping or "document_id" not in str(grouping):
        findings.append(
            {
                "field": "grouping_field",
                "expected": "document_id",
                "actual": grouping,
                "severity": "BLOCKER",
            }
        )
    return findings


DEFAULT_RAW_PATHS = {
    "tatoeba_cc0": "data/raw/external/tatoeba_cc0.tar.bz2",
    "qnl_arabic_ocr_v2": "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip",
    "wto_public_corpus_en_fr": "data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip",
}


def _check_sidecar(root, source_id, path):
    """Cheap consistency checks; full hash/license verification is opt-in."""
    sidecar = sidecar_path(path)
    if sidecar is None:
        return [f"{source_id}: missing provenance sidecar for {_relative(root, path)}"]
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"{source_id}: unreadable provenance sidecar: {exc}"]
    problems = []
    if meta.get("source_id") != source_id:
        problems.append(f"{source_id}: sidecar source_id is {meta.get('source_id')!r}")
    if meta.get("bytes") != path.stat().st_size:
        problems.append(f"{source_id}: sidecar bytes differ from file size")
    if not meta.get("sha256"):
        problems.append(f"{source_id}: sidecar has no sha256")
    return problems


def build_plan(root, *, output_dir, source_ids=None, raw_paths=None, config=None,
               verify_hashes=False, allow_partial=False):
    """Validate a future re-run before any corpus bytes are written."""
    from .pipeline import load_phase8_config

    root = Path(root).resolve()
    config = config or load_phase8_config()
    raw_paths = dict(DEFAULT_RAW_PATHS, **(raw_paths or {}))
    source_ids = list(source_ids or ["tatoeba_cc0", "qnl_arabic_ocr_v2", "wto_public_corpus_en_fr"])
    output = Path(output_dir)
    if output.is_absolute():
        try:
            output = output.resolve().relative_to(root)
        except ValueError:
            return {"ready": False, "blockers": ["output directory must stay inside the project root"], "sources": []}
    output = (root / output).resolve()
    blockers = []
    if output == (root / PHASE10_2_PROCESSED).resolve():
        blockers.append("refusing to target the frozen Phase 10.2 tree")
    if output.exists():
        blockers.append(f"output directory already exists: {_relative(root, output)}")

    sources = []
    for source_id in source_ids:
        try:
            entry = get_source(source_id)
        except ValueError as exc:
            blockers.append(str(exc))
            continue
        path = (root / raw_paths[source_id]).resolve()
        source = {
            "source_id": source_id,
            "format": entry.get("format"),
            "support": format_support(source_id),
            "path": _relative(root, path) if path.exists() else str(path),
            "exists": path.exists(),
            "license_id": entry.get("license_id"),
        }
        if not path.exists():
            blockers.append(f"{source_id}: raw artifact is missing: {source['path']}")
        else:
            blockers.extend(_check_sidecar(root, source_id, path))
            if verify_hashes:
                try:
                    source["sidecar"] = verify_download_provenance(source_id, path)
                except AcquisitionError as exc:
                    blockers.append(f"{source_id}: provenance verification failed: {exc}")
            if entry.get("format") == "zip_txt_with_official_metadata_csv":
                csv_candidate = path.with_name(f"{path.stem}_metadata.csv")
                if csv_candidate.is_file():
                    source["support"] = {
                        "source_id": source_id,
                        "format": entry.get("format"),
                        "support": "SUPPORTED",
                        "explanation": "corpus.adapters.qnl_adapter reads the QNL ZIP + metadata CSV and verifies embedded checksums.sha256.",
                    }
                    blockers.extend(_check_sidecar(root, source_id, csv_candidate))
                    if verify_hashes:
                        try:
                            source["metadata_csv_sidecar"] = verify_download_provenance(source_id, csv_candidate)
                        except AcquisitionError as exc:
                            blockers.append(f"{source_id}: metadata CSV provenance verification failed: {exc}")
            elif entry.get("format") == "zip_tab_delimited_parallel_corpus":
                import zipfile

                if zipfile.is_zipfile(path):
                    source["support"] = {
                        "source_id": source_id,
                        "format": entry.get("format"),
                        "support": "SUPPORTED",
                        "explanation": "corpus.adapters.wto_adapter streams the WTO ZIP TSV archive with continuation-line reconstruction and domain-stratified document sampling.",
                    }
        if source["support"]["support"] in {"UNSUPPORTED", "UNKNOWN"}:
            blockers.append(f"{source_id}: {source['support']['explanation']}")
        elif source["support"]["support"] == "PARTIAL":
            if allow_partial:
                source["limitation"] = source["support"]["explanation"]
            else:
                blockers.append(f"{source_id}: {source['support']['explanation']}")
        sources.append(source)

    approved_ratios = config["split_ratios"]
    approved_seed = config["random_seed"]
    plan = {
        "created_at_utc": _utc_now(),
        "root": str(root),
        "output_directory": _relative(root, output),
        "sources": sources,
        "approved_split_ratios": approved_ratios,
        "approved_seed": approved_seed,
        "verify_hashes": verify_hashes,
        "blockers": blockers,
        "ready": not blockers,
        "next_action": None,
    }
    if blockers:
        plan["next_action"] = "resolve every blocker; do not regenerate payloads before the plan is READY"
    else:
        plan["next_action"] = "run corpus.pipeline import/normalize, then data.splitting.split_sources into the new directory"
    return plan


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_run_manifest(
    root,
    *,
    run_id,
    config,
    sources,
    imports,
    normalizations,
    splits,
    split_decisions,
    adapter_versions=None,
    status="COMPLETE",
):
    """Build an auditable run manifest linking config, raw sources, stages, and splits."""
    import hashlib

    root = Path(root).resolve()
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(config, dict) or not config:
        raise ValueError("config must be a non-empty mapping")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    if not isinstance(imports, list) or not imports:
        raise ValueError("imports must be a non-empty list")
    if not isinstance(normalizations, list) or not normalizations:
        raise ValueError("normalizations must be a non-empty list")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must define train, validation, and test")

    config_canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    config_sha256 = hashlib.sha256(config_canonical.encode("utf-8")).hexdigest()

    source_entries = []
    for item in sources:
        raw_path = Path(item["raw_path"])
        if not raw_path.is_absolute():
            raw_path = (root / raw_path).resolve()
        sidecar = item.get("sidecar_path")
        if sidecar is None:
            sidecar = sidecar_path(raw_path)
        elif not Path(sidecar).is_absolute():
            sidecar = (root / sidecar).resolve()
        else:
            sidecar = Path(sidecar).resolve()
        source_entries.append(
            {
                "source_id": item["source_id"],
                "license_id": item["license_id"],
                "raw_path": _relative(root, raw_path),
                "raw_sha256": item.get("raw_sha256") or sha256_file(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "sidecar_path": _relative(root, sidecar) if sidecar and Path(sidecar).is_file() else None,
                "sidecar_sha256": sha256_file(sidecar) if sidecar and Path(sidecar).is_file() else None,
            }
        )

    import_entries = []
    for item in imports:
        imp_path = Path(item["imported_file"])
        if not imp_path.is_absolute():
            imp_path = (root / imp_path).resolve()
        meta_path = imp_path.with_name(imp_path.name + ".import.json")
        import_entries.append(
            {
                "source_id": item["source_id"],
                "imported_path": _relative(root, imp_path),
                "output_sha256": item.get("output_sha256") or sha256_file(imp_path),
                "bytes": imp_path.stat().st_size,
                "accepted_records": int(item["accepted_records"]),
                "source_sha256": item["source_sha256"],
                "adapter_version": item.get("adapter_version", "generic_v1"),
                "sidecar_path": _relative(root, meta_path) if meta_path.is_file() else None,
                "sidecar_sha256": sha256_file(meta_path) if meta_path.is_file() else None,
            }
        )

    norm_entries = []
    for item in normalizations:
        norm_path = Path(item["output_file"])
        if not norm_path.is_absolute():
            norm_path = (root / norm_path).resolve()
        in_path = Path(item["input_file"])
        if not in_path.is_absolute():
            in_path = (root / in_path).resolve()
        meta_path = norm_path.with_name(norm_path.name + ".normalized.json")
        norm_entries.append(
            {
                "source_id": item.get("source_id"),
                "input_path": _relative(root, in_path),
                "input_sha256": item.get("input_sha256") or sha256_file(in_path),
                "normalized_path": _relative(root, norm_path),
                "output_sha256": item.get("output_sha256") or sha256_file(norm_path),
                "bytes": norm_path.stat().st_size,
                "accepted_records": int(item["accepted_records"]),
                "retain_stage_texts": bool(item.get("retain_stage_texts", True)),
                "sidecar_path": _relative(root, meta_path) if meta_path.is_file() else None,
                "sidecar_sha256": sha256_file(meta_path) if meta_path.is_file() else None,
            }
        )

    split_entries = {}
    for split_name, info in splits.items():
        sp_path = Path(info["path"] if isinstance(info, dict) else info)
        if not sp_path.is_absolute():
            sp_path = (root / sp_path).resolve()
        records_count = info.get("records") if isinstance(info, dict) else None
        if records_count is None:
            with sp_path.open("rb") as stream:
                records_count = sum(1 for line in stream if line.strip())
        split_entries[split_name] = {
            "path": _relative(root, sp_path),
            "sha256": (info.get("sha256") if isinstance(info, dict) else None) or sha256_file(sp_path),
            "bytes": sp_path.stat().st_size,
            "records": int(records_count),
        }

    decision_findings = check_split_decisions(split_decisions, config)
    return {
        "schema_version": 1,
        "run_id": run_id.strip(),
        "created_at_utc": _utc_now(),
        "status": status if not decision_findings else "NON_CONFORMING_SPLITS",
        "root": str(root),
        "config_sha256": config_sha256,
        "adapter_versions": dict(adapter_versions or {}),
        "sources": source_entries,
        "imports": import_entries,
        "normalizations": norm_entries,
        "splits": split_entries,
        "split_decisions": split_decisions,
        "split_decision_findings": decision_findings,
    }


def verify_run_manifest(root, manifest_or_path, *, config=None, verify_hashes=True):
    """Verify every file, sidecar link, SHA-256, and split decision in a run manifest."""
    root = Path(root).resolve()
    if isinstance(manifest_or_path, (str, os.PathLike, Path)):
        mpath = Path(manifest_or_path)
        if not mpath.is_absolute():
            mpath = (root / mpath).resolve()
        if not mpath.is_file():
            return {"valid": False, "errors": [f"run manifest not found: {mpath}"]}
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    elif isinstance(manifest_or_path, dict):
        manifest = manifest_or_path
    else:
        return {"valid": False, "errors": ["manifest_or_path must be a path or dict"]}

    errors = []
    if manifest.get("schema_version") != 1:
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("status") != "COMPLETE":
        errors.append(f"run manifest status is {manifest.get('status')!r}, expected 'COMPLETE'")

    if config is not None:
        import hashlib

        expected_cfg = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("config_sha256") != expected_cfg:
            errors.append("config_sha256 does not match active configuration")
        findings = check_split_decisions(manifest.get("split_decisions") or {}, config)
        if findings:
            errors.extend(f"split decision blocker on {f['field']}: expected {f['expected']}, got {f['actual']}" for f in findings)

    raw_hashes = {}
    for src in manifest.get("sources") or []:
        sid = src.get("source_id")
        rpath = (root / src.get("raw_path", "")).resolve()
        if not rpath.is_file():
            errors.append(f"{sid}: missing raw file {src.get('raw_path')}")
            continue
        if verify_hashes and sha256_file(rpath) != src.get("raw_sha256"):
            errors.append(f"{sid}: raw_sha256 mismatch for {src.get('raw_path')}")
        raw_hashes[sid] = src.get("raw_sha256")
        if not src.get("sidecar_path") or not (root / src["sidecar_path"]).is_file():
            errors.append(f"{sid}: missing raw provenance sidecar")

    import_hashes = {}
    for imp in manifest.get("imports") or []:
        sid = imp.get("source_id")
        ipath = (root / imp.get("imported_path", "")).resolve()
        if not ipath.is_file():
            errors.append(f"{sid}: missing imported file {imp.get('imported_path')}")
            continue
        if raw_hashes.get(sid) and imp.get("source_sha256") != raw_hashes.get(sid):
            errors.append(f"{sid}: import source_sha256 does not match raw source sha256")
        if verify_hashes and sha256_file(ipath) != imp.get("output_sha256"):
            errors.append(f"{sid}: imported output_sha256 mismatch for {imp.get('imported_path')}")
        import_hashes[imp.get("imported_path")] = imp.get("output_sha256")
        if not imp.get("sidecar_path") or not (root / imp["sidecar_path"]).is_file():
            errors.append(f"{sid}: missing .import.json sidecar for {imp.get('imported_path')}")

    for norm in manifest.get("normalizations") or []:
        npath = (root / norm.get("normalized_path", "")).resolve()
        if not npath.is_file():
            errors.append(f"missing normalized file {norm.get('normalized_path')}")
            continue
        expected_in_sha = import_hashes.get(norm.get("input_path"))
        if expected_in_sha and norm.get("input_sha256") != expected_in_sha:
            errors.append(f"normalized input_sha256 mismatch with imported output for {norm.get('normalized_path')}")
        if verify_hashes and sha256_file(npath) != norm.get("output_sha256"):
            errors.append(f"normalized output_sha256 mismatch for {norm.get('normalized_path')}")
        if not norm.get("sidecar_path") or not (root / norm["sidecar_path"]).is_file():
            errors.append(f"missing .normalized.json sidecar for {norm.get('normalized_path')}")

    splits = manifest.get("splits") or {}
    if set(splits) != {"train", "validation", "test"}:
        errors.append("manifest splits must contain train, validation, and test")
    for sname, sinfo in splits.items():
        spath = (root / sinfo.get("path", "")).resolve()
        if not spath.is_file():
            errors.append(f"missing split file for {sname}: {sinfo.get('path')}")
            continue
        if verify_hashes and sha256_file(spath) != sinfo.get("sha256"):
            errors.append(f"split sha256 mismatch for {sname}: {sinfo.get('path')}")
        if int(sinfo.get("records", 0)) <= 0:
            errors.append(f"split {sname} has zero records")

    return {"valid": not errors, "errors": errors, "run_id": manifest.get("run_id")}



def render_markdown(audit):
    lines = [
        "# Phase 10.2 artifact audit",
        "",
        f"Generated: `{audit['created_at_utc']}`",
        f"Artifacts: **{audit['summary']['artifacts']}** — bytes: **{audit['summary']['bytes']:,}**",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    for verdict, count in sorted(audit["summary"]["by_reproducibility"].items()):
        lines.append(f"| {verdict} | {count} |")
    lines += ["", "## Payload artifacts", "", "| Path | Bytes | SHA-256 | Reproducible | Reason |", "| --- | ---: | --- | --- | --- |"]
    for item in audit["artifacts"]:
        if item["kind"] in {"report"}:
            continue
        digest = (item["sha256"] or "not computed")[:16]
        reason = item["reason"].replace("|", "\\|")
        lines.append(f"| `{item['path']}` | {item['bytes']:,} | `{digest}` | {item['reproducible']} | {reason} |")
    lines.append("")
    lines.append("Report files under `reports/phase10/phase10_2/` are evidence, not payload, and are listed in the JSON.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m corpus.reproducibility")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="inventory Phase 10.2 artifacts and hashes")
    audit.add_argument("--root", default=".")
    audit.add_argument("--json", default=str(PHASE10_2_REPORTS / "ARTIFACT_AUDIT.json"))
    audit.add_argument("--markdown", default=str(PHASE10_2_REPORTS / "ARTIFACT_AUDIT.md"))
    audit.add_argument("--no-hash", action="store_true")

    plan = sub.add_parser("plan", help="validate a re-run before writing corpus bytes")
    plan.add_argument("--root", default=".")
    plan.add_argument("--output", required=True)
    plan.add_argument("--source", action="append", dest="sources")
    plan.add_argument("--verify-hashes", action="store_true")
    plan.add_argument("--allow-partial", action="store_true")
    plan.add_argument("--json", default=None)

    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "audit":
        result = audit_artifacts(root, hash_files=not args.no_hash)
        write_json_atomic(root / args.json, result)
        (root / args.markdown).write_text(render_markdown(result), encoding="utf-8")
        print(json.dumps({"json": args.json, "markdown": args.markdown, "summary": result["summary"]}, indent=2))
        return 0
    result = build_plan(root, output_dir=args.output, source_ids=args.sources,
                        verify_hashes=args.verify_hashes, allow_partial=args.allow_partial)
    if args.json:
        write_json_atomic(root / args.json, result)
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
