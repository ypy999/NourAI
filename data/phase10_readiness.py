"""Configurable, fail-closed corpus readiness assessment for Phase 10."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist


STATES = {"READY", "NOT_READY", "BLOCKED", "UNKNOWN"}


def load_readiness_policy(path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "phase10_readiness.json"
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load Phase 10 readiness policy: {exc}") from exc
    validate_readiness_policy(policy)
    return policy


def validate_readiness_policy(policy):
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("readiness policy must be an object with schema_version=1")
    languages = policy.get("target_languages")
    if not isinstance(languages, list) or not languages or any(not isinstance(item, str) or not item for item in languages):
        raise ValueError("target_languages must be a nonempty list of language names")
    sampling = policy.get("statistical_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("statistical_sampling must be an object")
    confidence = sampling.get("confidence_level")
    margin = sampling.get("maximum_margin_of_error")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 < margin < 0.5:
        raise ValueError("maximum_margin_of_error must be between 0 and 0.5")
    if sampling.get("sampling_unit") != "records":
        raise ValueError("sampling_unit must be 'records'")
    diversity = policy.get("diversity")
    if not isinstance(diversity, dict):
        raise ValueError("diversity must be an object")
    for key in ("minimum_unique_sources_per_language", "minimum_verified_domains_per_language"):
        value = diversity.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    size = policy.get("dataset_size")
    if not isinstance(size, dict):
        raise ValueError("dataset_size must be an object")
    floor = size.get("minimum_train_records_per_language")
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, int) or floor < 1):
        raise ValueError("minimum_train_records_per_language must be null or a positive integer")
    return policy


def required_sample_size(population, *, confidence_level, margin_of_error, finite_population_correction=True):
    """Worst-case proportion sample plan; returns None when population is unknown."""
    if isinstance(population, bool) or not isinstance(population, int) or population < 0:
        raise ValueError("population must be a non-negative integer")
    if population == 0:
        return 0
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    uncorrected = z * z * 0.25 / (margin_of_error * margin_of_error)
    if finite_population_correction:
        required = math.ceil(uncorrected / (1.0 + (uncorrected - 1.0) / population))
    else:
        required = math.ceil(uncorrected)
    return min(population, required)


def _check(state, details=None, blockers=None):
    return {"status": state, "details": details or {}, "blockers": blockers or []}


def _count(mapping, key):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def assess_phase10_readiness(*, language_statistics, split_language_statistics,
                             language_diversity, used_sources, split_validation,
                             quality_statistics, tokenizer_compatibility, policy=None):
    """Assess independent readiness criteria and never coerce missing evidence to READY.

    `split_language_statistics` is keyed by language, then split name, with
    `{records: int}` values. `language_diversity` is keyed by language and
    requires `unique_sources` and `verified_domains` counts.
    """
    policy = validate_readiness_policy(policy or load_readiness_policy())
    languages = policy["target_languages"]
    confidence = policy["statistical_sampling"]["confidence_level"]
    margin = policy["statistical_sampling"]["maximum_margin_of_error"]
    fpc = policy["statistical_sampling"].get("finite_population_correction", True)
    min_sources = policy["diversity"]["minimum_unique_sources_per_language"]
    min_domains = policy["diversity"]["minimum_verified_domains_per_language"]
    train_floor = policy["dataset_size"]["minimum_train_records_per_language"]

    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "policy": policy, "languages": {}, "checks": {}, "training_started": False}

    if used_sources is None:
        result["checks"]["legal_readiness"] = _check("UNKNOWN", blockers=["source/license evidence is missing"])
    elif not isinstance(used_sources, (list, tuple)):
        raise ValueError("used_sources must be a list or null")
    elif not used_sources:
        result["checks"]["legal_readiness"] = _check("UNKNOWN", blockers=["no source/license evidence was supplied"])
    else:
        legal_blockers, legal_unknown = [], []
        for source in used_sources:
            if not isinstance(source, dict):
                legal_unknown.append("source entry is not an object")
                continue
            source_id = source.get("source_id", "unknown")
            state = source.get("state", source.get("license_status", source.get("status")))
            approved = source.get("approved_for_ingestion")
            if state is None or approved is None:
                legal_unknown.append(f"{source_id}: legal status/approval is missing")
            elif state != "VERIFIED" or approved is not True:
                legal_blockers.append(f"{source_id}: not verified and approved for ingestion")
        legal_state = "BLOCKED" if legal_blockers else "UNKNOWN" if legal_unknown else "READY"
        result["checks"]["legal_readiness"] = _check(legal_state, {"source_count": len(used_sources)}, legal_blockers + legal_unknown)

    validation = split_validation
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        if not isinstance(validation, dict):
            result["checks"]["document_leakage"] = _check("UNKNOWN", blockers=["split validation evidence is missing"])
            result["checks"]["duplicate_leakage"] = _check("UNKNOWN", blockers=["split validation evidence is missing"])
        else:
            result["checks"]["document_leakage"] = _check("BLOCKED" if validation.get("document_group_leakage", 0) else "UNKNOWN",
                blockers=validation.get("errors", ["split validation did not pass"]))
            result["checks"]["duplicate_leakage"] = _check("BLOCKED" if validation.get("duplicate_leakage", 0) or validation.get("duplicates_within_splits", 0) else "UNKNOWN",
                blockers=validation.get("errors", ["split validation did not pass"]))
    else:
        group_leaks = validation.get("document_group_leakage")
        duplicate_leaks = validation.get("duplicate_leakage")
        duplicate_internal = validation.get("duplicates_within_splits")
        result["checks"]["document_leakage"] = _check(
            "UNKNOWN" if group_leaks is None else "BLOCKED" if group_leaks else "READY",
            {"leaked_groups": group_leaks}, ["document group leakage detected"] if group_leaks else (["document leakage metric missing"] if group_leaks is None else []))
        duplicate_missing = duplicate_leaks is None or duplicate_internal is None
        duplicate_bad = bool(duplicate_leaks) or bool(duplicate_internal)
        result["checks"]["duplicate_leakage"] = _check(
            "BLOCKED" if duplicate_bad else "UNKNOWN" if duplicate_missing else "READY",
            {"cross_split_duplicates": duplicate_leaks, "within_split_duplicates": duplicate_internal},
            ["duplicate leakage detected"] if duplicate_bad else (["duplicate metrics missing"] if duplicate_missing else []))

    if not isinstance(quality_statistics, dict):
        result["checks"]["data_quality"] = _check("UNKNOWN", blockers=["quality report is missing"])
    else:
        required_quality = ("records_checked", "empty_records", "invalid_unicode_records", "corrupted_records")
        absent = [key for key in required_quality if _count(quality_statistics, key) is None]
        bad = any(quality_statistics.get(key, 0) for key in ("empty_records", "invalid_unicode_records", "corrupted_records"))
        state = "BLOCKED" if bad else "UNKNOWN" if absent else "READY"
        result["checks"]["data_quality"] = _check(state, quality_statistics,
            (["structural/content corruption indicators were reported"] if bad else []) + (["required quality metrics are missing: " + ", ".join(absent)] if absent else []))

    if not isinstance(tokenizer_compatibility, dict):
        result["checks"]["tokenizer_compatibility"] = _check("UNKNOWN", blockers=["tokenizer compatibility evidence is missing"])
    else:
        compatible = tokenizer_compatibility.get("compatible")
        state = "UNKNOWN" if compatible is None else "READY" if compatible is True else "BLOCKED"
        result["checks"]["tokenizer_compatibility"] = _check(state, tokenizer_compatibility,
            [] if state == "READY" else ["tokenizer compatibility is false" if state == "BLOCKED" else "tokenizer compatibility was not evaluated"])

    per_language_states = {}
    for language in languages:
        stats = (language_statistics or {}).get(language, {})
        splits = (split_language_statistics or {}).get(language, {})
        diversity = (language_diversity or {}).get(language, {})
        total_records = _count(stats, "records") if isinstance(stats, dict) else None
        train_records = _count(splits.get("train", {}), "records") if isinstance(splits, dict) else None
        validation_records = _count(splits.get("validation", {}), "records") if isinstance(splits, dict) else None
        test_records = _count(splits.get("test", {}), "records") if isinstance(splits, dict) else None
        language_checks = {}
        language_checks["language_coverage"] = _check(
            "UNKNOWN" if total_records is None else "READY" if total_records > 0 else "NOT_READY",
            {"records": total_records}, [] if total_records else (["language has no records"] if total_records == 0 else ["language record count is missing"]))

        if total_records is None or validation_records is None or test_records is None:
            sample_status = "UNKNOWN"
            sample_blockers = ["total, validation, or test record counts are missing"]
            required = None
        else:
            required = required_sample_size(total_records, confidence_level=confidence,
                margin_of_error=margin, finite_population_correction=fpc)
            insufficient = validation_records < required or test_records < required
            sample_status = "NOT_READY" if insufficient else "READY"
            sample_blockers = ([f"validation/test records are below the planned sample size of {required}"] if insufficient else [])
        language_checks["validation_reliability"] = _check(sample_status,
            {"records": validation_records, "required_records": required, "confidence_level": confidence,
             "maximum_margin_of_error": margin, "planning_assumption": "worst-case proportion p=0.5; simple random sampling"}, sample_blockers)
        language_checks["test_reliability"] = _check(sample_status,
            {"records": test_records, "required_records": required, "confidence_level": confidence,
             "maximum_margin_of_error": margin, "planning_assumption": "worst-case proportion p=0.5; simple random sampling"}, sample_blockers)

        source_count = _count(diversity, "unique_sources") if isinstance(diversity, dict) else None
        domain_count = _count(diversity, "verified_domains") if isinstance(diversity, dict) else None
        source_state = "UNKNOWN" if source_count is None else "READY" if source_count >= min_sources else "NOT_READY"
        domain_state = "UNKNOWN" if domain_count is None else "READY" if domain_count >= min_domains else "NOT_READY"
        language_checks["source_diversity"] = _check(source_state,
            {"unique_sources": source_count, "required": min_sources},
            ["source count is missing"] if source_count is None else ([f"requires {min_sources} unique sources"] if source_state != "READY" else []))
        language_checks["domain_coverage"] = _check(domain_state,
            {"verified_domains": domain_count, "required": min_domains},
            ["verified domain count is missing"] if domain_count is None else ([f"requires {min_domains} verified domains"] if domain_state != "READY" else []))

        if train_floor is None or train_records is None:
            size_state = "UNKNOWN"
            size_blockers = ["a justified training-size policy or train count is missing"]
        else:
            size_state = "READY" if train_records >= train_floor else "NOT_READY"
            size_blockers = [] if size_state == "READY" else [f"train records are below configured floor {train_floor}"]
        language_checks["dataset_size"] = _check(size_state,
            {"train_records": train_records, "configured_minimum": train_floor}, size_blockers)
        result["languages"][language] = {"status": _combine([item["status"] for item in language_checks.values()]),
                                          "checks": language_checks}
        per_language_states[language] = result["languages"][language]["status"]

    result["overall_status"] = _combine(
        [item["status"] for item in result["checks"].values()] + list(per_language_states.values()))
    result["language_statuses"] = per_language_states
    result["assumptions_and_limits"] = [
        "Sample-size planning assumes independent simple random samples and a worst-case proportion; it is not a guarantee of validation/test model-metric precision.",
        "Heuristic domains, language labels, and surface quality checks are not semantic or factual verification.",
        "Missing evidence remains UNKNOWN and can never be counted as READY.",
    ]
    return result


def _combine(states):
    if not states:
        return "UNKNOWN"
    if "BLOCKED" in states:
        return "BLOCKED"
    if "NOT_READY" in states:
        return "NOT_READY"
    if "UNKNOWN" in states:
        return "UNKNOWN"
    return "READY" if all(state == "READY" for state in states) else "UNKNOWN"
