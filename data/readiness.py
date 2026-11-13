"""Three-state technical and training-readiness assessment."""
from corpus.readiness import assess_readiness

READY, CONDITIONALLY_READY, NOT_READY = "READY", "CONDITIONALLY_READY", "NOT_READY"


def assess_corpus(*, technical_checks, license_statuses, language_coverage=None, other_blockers=()):
    blockers = list(other_blockers)
    if not technical_checks.get("valid", False):
        blockers.append("one or more data integrity checks failed")
    if any(state != "VERIFIED" for state in license_statuses):
        blockers.append("one or more source licenses are not verified")
    if blockers:
        status = NOT_READY
    elif language_coverage:
        missing = [name for name, value in language_coverage.items() if value in (0, None)]
        status = CONDITIONALLY_READY if missing else READY
        if missing:
            blockers.append("missing requested language coverage: " + ", ".join(missing))
    else:
        status = READY
    return {"status": status, "technical_checks": technical_checks,
            "legal_statuses": list(license_statuses), "blockers": blockers,
            "training_ready": False, "size_threshold_used": False}


__all__ = ["READY", "CONDITIONALLY_READY", "NOT_READY", "assess_corpus", "assess_readiness"]
