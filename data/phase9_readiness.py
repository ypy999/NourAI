"""Separate legal, technical, linguistic, diversity and training readiness gates."""
from datetime import datetime, timezone

from .language_balancing import LANGUAGE_GROUPS


def _state(ok, *, conditional=False):
    return "READY" if ok else "CONDITIONALLY_READY" if conditional else "NOT_READY"


def assess_phase9_readiness(*, language_statistics, split_language_statistics,
                            used_sources, split_validation, balance_statistics,
                            min_eval_records=100, min_eval_tokens=10_000,
                            min_sources_per_language=2, min_verified_domains_per_language=3,
                            training_allowed=False):
    """Assess readiness from observed data; corpus size alone is never sufficient.

    Evaluation coverage thresholds are explicit, configurable guardrails for
    independent held-out sets; they do not imply statistically definitive
    model evaluation or language competence.
    """
    for name, value in (("min_eval_records", min_eval_records), ("min_eval_tokens", min_eval_tokens),
                        ("min_sources_per_language", min_sources_per_language),
                        ("min_verified_domains_per_language", min_verified_domains_per_language)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    blockers = {"legal": [], "technical": [], "linguistic": [], "diversity": [], "training": []}
    source_errors = []
    for source in used_sources:
        if source.get("status") != "VERIFIED" or not source.get("approved_for_ingestion", False):
            source_errors.append(f"{source.get('source_id', 'unknown')} is not verified and approved for ingestion")
        if source.get("license_id") in (None, "UNKNOWN_LICENSE", "unknown"):
            source_errors.append(f"{source.get('source_id', 'unknown')} has no known license")
    blockers["legal"].extend(source_errors)
    legal = _state(not source_errors)

    validation_ok = bool(split_validation and split_validation.get("valid") is True
                         and split_validation.get("duplicate_leakage") == 0
                         and split_validation.get("duplicates_within_splits", 0) == 0)
    if not validation_ok:
        blockers["technical"].extend((split_validation or {}).get("errors", []) or ["split validation or duplicate-leakage check did not pass"])
    technical = _state(validation_ok)

    per_language = {}
    all_eval_present = True
    all_eval_sufficient = True
    for language in ("arabic", "english", "french"):
        overall = language_statistics.get(language, {})
        language_splits = split_language_statistics.get(language, {})
        details = {"corpus_records": int(overall.get("records", 0)),
                   "corpus_tokens": int(overall.get("tokens_including_eos", 0)),
                   "validation_records": int(language_splits.get("validation_records", 0)),
                   "validation_tokens": int(language_splits.get("validation_tokens", 0)),
                   "test_records": int(language_splits.get("test_records", 0)),
                   "test_tokens": int(language_splits.get("test_tokens", 0))}
        present = details["corpus_records"] > 0 and details["validation_records"] > 0 and details["test_records"] > 0
        sufficient = (details["validation_records"] >= min_eval_records and details["test_records"] >= min_eval_records
                      and details["validation_tokens"] >= min_eval_tokens and details["test_tokens"] >= min_eval_tokens)
        details["independent_evaluation_present"] = present
        details["meets_configured_evaluation_floor"] = sufficient
        per_language[language] = details
        all_eval_present &= present
        all_eval_sufficient &= sufficient
        if not present:
            blockers["linguistic"].append(f"{language} is absent from the corpus or an independent validation/test split")
        elif not sufficient:
            blockers["linguistic"].append(f"{language} validation/test coverage is below the configured floor ({min_eval_records} records and {min_eval_tokens} tokens per split)")
    linguistic = _state(all_eval_sufficient, conditional=all_eval_present)

    balance_languages = (balance_statistics or {}).get("languages", {})
    for language in ("arabic", "english", "french"):
        stats = balance_languages.get(language, {})
        source_count = int(stats.get("unique_sources", 0))
        verified_domain_count = int(stats.get("verified_domain_count", 0))
        if source_count < min_sources_per_language:
            blockers["diversity"].append(f"{language} has {source_count} source(s); configured minimum is {min_sources_per_language}")
        if verified_domain_count < min_verified_domains_per_language:
            blockers["diversity"].append(f"{language} has {verified_domain_count} source-verified domain(s); configured minimum is {min_verified_domains_per_language}")
    diversity = _state(not blockers["diversity"], conditional=all_eval_present)

    if not training_allowed:
        blockers["training"].append("training is explicitly out of scope for Phase 9")
    if legal != "READY" or technical != "READY" or linguistic != "READY" or diversity != "READY":
        blockers["training"].append("one or more legal, technical, language-evaluation, or diversity gates are not ready")
    training = _state(training_allowed and legal == technical == linguistic == diversity == "READY")
    statuses = (legal, technical, linguistic, diversity)
    overall = "NOT_READY" if "NOT_READY" in statuses else "CONDITIONALLY_READY" if "CONDITIONALLY_READY" in statuses else "READY"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "legal_readiness": {"status": legal, "blockers": blockers["legal"], "used_source_count": len(used_sources)},
        "technical_readiness": {"status": technical, "blockers": blockers["technical"], "split_validation": split_validation},
        "linguistic_readiness": {"status": linguistic, "languages": per_language,
                                 "configured_minimum_eval_records_per_split": min_eval_records,
                                 "configured_minimum_eval_tokens_per_split": min_eval_tokens,
                                 "blockers": blockers["linguistic"]},
        "diversity_readiness": {"status": diversity, "minimum_sources_per_language": min_sources_per_language,
                                "minimum_source_verified_domains_per_language": min_verified_domains_per_language,
                                "blockers": blockers["diversity"]},
        "training_readiness": {"status": training, "training_started": False,
                               "blockers": blockers["training"]},
        "all_blockers": [message for group in blockers.values() for message in group],
        "size_alone_used_as_readiness_gate": False,
        "note": "Evaluation floors are configurable operational guardrails, not claims of statistically sufficient model evaluation. Domain heuristics and source metadata are not independent semantic verification.",
    }
