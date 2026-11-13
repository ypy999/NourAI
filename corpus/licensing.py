"""Explicit provenance and license policy. A catalog entry is not legal advice."""

import json
from pathlib import Path

SOURCES = {
    "tatoeba_cc0": {
        "name": "Tatoeba sentences CC0 export",
        "url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
        "format": "tatoeba_cc0_tsv_bz2",
        "license_id": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_license_url": "https://tatoeba.org/en/downloads",
        "status": "VERIFIED",
        "approved_for_ingestion": True,
        "languages": ["ara", "eng", "fra", "multilingual"],
        "approximate_size_bytes": 7963537,
        "content_type": "sentence-level TSV in tar.bz2",
        "release_version": "official sentences_CC0 export; upstream date is recorded in archive/source listing",
        "license_version": "CC0-1.0",
        "commercial_use": "permitted by CC0 terms",
        "attribution_required": False,
        "sharealike_required": False,
        "derivatives": "permitted by CC0 terms",
        "geographic_restrictions": None,
        "temporal_restrictions": None,
        "checked_at": "2026-11-11",
        "verification_method": "official Tatoeba CC0-specific export page and official CC0 legal code; checksum and archive layout verified locally",
        "legal_review_note": "Source/label verification, not individualized legal advice or independent counsel opinion.",
        "allowed_use": "corpus ingestion for the explicitly CC0 export only",
        "notes": "Do not substitute the general sentences export; it includes CC BY content.",
    },
    "tatoeba_general": {
        "name": "Tatoeba general sentences export",
        "url": "https://downloads.tatoeba.org/exports/sentences.tar.bz2",
        "format": "tsv_bz2",
        "license_id": "CC-BY-2.0-FR",
        "license_url": "https://creativecommons.org/licenses/by/2.0/fr/",
        "source_license_url": "https://tatoeba.org/en/terms_of_use",
        "status": "NEEDS_REVIEW",
        "approved_for_ingestion": False,
        "languages": ["multilingual"],
        "approximate_size_bytes": None,
        "content_type": "sentence-level TSV",
        "release_version": None,
        "license_version": "CC-BY-2.0-FR",
        "commercial_use": "requires review of applicable terms",
        "attribution_required": True,
        "sharealike_required": False,
        "derivatives": "requires review of applicable terms",
        "geographic_restrictions": None,
        "temporal_restrictions": None,
        "checked_at": "2026-11-11",
        "verification_method": "official source terms page reviewed; no ingestion performed",
        "legal_review_note": "Attribution and corpus use require review.",
        "allowed_use": None,
        "notes": "Attribution obligations must be evaluated before corpus use.",
    },
    "wikimedia": {
        "name": "Wikimedia Wikipedia dumps",
        "url": "https://dumps.wikimedia.org/",
        "format": "xml_bz2",
        "license_id": "CC-BY-SA-4.0/GFDL (project and revision dependent)",
        "license_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
        "source_license_url": "https://meta.wikimedia.org/wiki/Terms_of_Use",
        "status": "NEEDS_REVIEW",
        "approved_for_ingestion": False,
        "languages": ["multilingual"],
        "approximate_size_bytes": None,
        "content_type": "encyclopedic XML dump",
        "release_version": "rolling latest; project dump date must be pinned before use",
        "license_version": "project/revision dependent",
        "commercial_use": "subject to project terms and license obligations",
        "attribution_required": True,
        "sharealike_required": True,
        "derivatives": "share-alike obligations may apply",
        "geographic_restrictions": None,
        "temporal_restrictions": None,
        "checked_at": "2026-11-11",
        "verification_method": "official Wikimedia terms reviewed; per-revision validation not completed",
        "legal_review_note": "Needs project/revision-level license and attribution review.",
        "allowed_use": None,
        "notes": "Attribution/share-alike and per-project/revision conditions require review.",
    },
    "oscar": {
        "name": "OSCAR Common Crawl derived corpora",
        "url": "https://huggingface.co/datasets/oscar",
        "format": "parquet_or_jsonl",
        "license_id": "Underlying page licenses vary / are not established by packaging license",
        "license_url": None,
        "source_license_url": "https://huggingface.co/datasets/oscar",
        "status": "UNKNOWN_LICENSE",
        "approved_for_ingestion": False,
        "languages": ["multilingual"],
        "approximate_size_bytes": None,
        "content_type": "web crawl text corpus",
        "release_version": "dataset card version must be pinned before use",
        "license_version": "underlying source rights vary or unknown",
        "commercial_use": "unknown",
        "attribution_required": "unknown",
        "sharealike_required": "unknown",
        "derivatives": "unknown",
        "geographic_restrictions": "unknown",
        "temporal_restrictions": "unknown",
        "checked_at": "2026-11-11",
        "verification_method": "dataset card reviewed; underlying page-level rights not established",
        "legal_review_note": "Do not ingest automatically.",
        "allowed_use": None,
        "notes": "Dataset packaging terms do not establish rights for each underlying web page.",
    },
}

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_sources.json"
try:
    SOURCES = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"could not load dataset source registry: {_REGISTRY_PATH}") from exc
if not isinstance(SOURCES, dict):
    raise RuntimeError("dataset source registry root must be a JSON object")


def get_source(source_id):
    try:
        return dict(SOURCES[source_id])
    except KeyError as exc:
        raise ValueError(f"unknown corpus source: {source_id}") from exc


def source_use_status(source_id):
    """Only the explicitly CC0 export is eligible for automatic ingestion."""
    source = get_source(source_id)
    return {
        "source_id": source_id,
        "status": source["status"],
        "approved_for_automatic_ingestion": source["status"] == "VERIFIED" and source.get("approved_for_ingestion", False),
        "license_id": source["license_id"],
        "license_url": source["license_url"],
        "notes": source["notes"],
    }


def require_approved_source(source_id):
    source = get_source(source_id)
    if source["status"] != "VERIFIED" or not source.get("approved_for_ingestion", False):
        raise PermissionError(f"source {source_id!r} is {source['status']}; manual license review is required")
    return source
