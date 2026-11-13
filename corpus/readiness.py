"""Evidence-based corpus readiness; data size is descriptive, never a gate."""
from datetime import datetime, timezone


def assess_readiness(*, sources, imports, splits, quality, validation, errors=(), language_coverage=None):
    blockers = []
    if not sources:
        blockers.append("no source provenance registered")
    for source in sources:
        state = source.get("status", source.get("state"))
        if state != "VERIFIED" or not source.get("approved_for_ingestion", False):
            blockers.append(f"source {source.get('source_id', 'unknown')} is not approved for automatic corpus ingestion")
        if not source.get("license_id") or source.get("license_id") in ("UNKNOWN", "unknown"):
            blockers.append("source license is missing or unknown")
    if not imports or sum(int(item.get("accepted_records", 0)) for item in imports) == 0:
        blockers.append("no text records were imported")
    if any(item.get("errors") or item.get("errors_truncated", 0) for item in imports):
        blockers.append("one or more imports contain parse/encoding errors")
    if not splits or any(int(splits.get(name, 0)) <= 0 for name in ("train", "validation", "test")):
        blockers.append("one or more deduplicated data splits are empty")
    if errors:
        blockers.append("source validation/import errors require review")
    if validation and validation.get("valid") is False:
        blockers.append("dataset validation did not pass")
    missing_languages = [name for name, count in (language_coverage or {}).items() if count in (0, None)]
    status = "NOT_READY" if blockers else "CONDITIONALLY_READY" if missing_languages else "READY"
    if missing_languages:
        blockers.append("requested language(s) have no coverage: " + ", ".join(missing_languages))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "technical_status": "PASS" if validation and validation.get("valid") and not errors else "FAIL",
        "legal_status": "VERIFIED" if sources and all(source.get("status", source.get("state")) == "VERIFIED" for source in sources) else "NEEDS_REVIEW",
        "training_started": False,
        "cuda_used": False,
        "size_threshold_used": False,
        "blockers": blockers,
        "data_quality_score": quality,
        "validation": validation,
        "source_count": len(sources or []),
        "imports": imports or [],
        "split_counts": splits or {"train": 0, "validation": 0, "test": 0},
        "note": "Technical readiness is not legal advice, evidence of language understanding, or a model quality claim.",
    }
