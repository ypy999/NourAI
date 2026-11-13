"""Compatibility checks for license-aware corpus composition."""


class LicenseCompatibilityError(PermissionError):
    pass


def check_corpus_licenses(sources):
    sources = list(sources)
    blockers = []
    for source in sources:
        if source.get("status", source.get("state")) != "VERIFIED" or not source.get("approved_for_ingestion"):
            blockers.append(f"{source.get('source_id', 'unknown')}: license is not verified for ingestion")
        if not source.get("license_id") or str(source.get("license_id")).casefold().startswith("unknown"):
            blockers.append(f"{source.get('source_id', 'unknown')}: license is unknown")
    license_ids = {source.get("license_id") for source in sources if source.get("license_id")}
    if len(license_ids) > 1:
        blockers.append("mixed license identifiers require an explicit compatibility/manual review")
    return {"compatible": not blockers, "license_ids": sorted(str(item) for item in license_ids), "blockers": blockers}


def require_compatible(sources):
    result = check_corpus_licenses(sources)
    if not result["compatible"]:
        raise LicenseCompatibilityError("; ".join(result["blockers"]))
    return result
