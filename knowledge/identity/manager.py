"""Jebrex AI — Central Identity Registry Loader & Versioned Self-Knowledge Manager (Phase I)."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_IDENTITY_PATH = Path(__file__).resolve().parent / "jebrex_identity.json"

_REQUIRED_TOP_KEYS = (
    "schema_version",
    "identity_version",
    "official_identity",
    "ownership_and_development",
    "architecture",
    "languages_supported",
    "verified_capabilities",
    "capabilities_in_development",
    "known_limitations",
    "canonical_responses",
    "version_history",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_identity_record(record: Dict[str, Any]) -> None:
    """Validate structural integrity and independence invariants of the central identity registry."""
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in record]
    if missing:
        raise ValueError(f"Identity registry missing required sections: {missing}")

    ident = record["official_identity"]
    if ident.get("name") != "Jebrex AI":
        raise ValueError("Official identity name must be 'Jebrex AI'")

    indep = ident.get("independence_facts", {})
    if not indep.get("is_independent_project", False):
        raise ValueError("Jebrex AI must be marked as an independent project")
    if indep.get("affiliated_with_openai") or indep.get("affiliated_with_google_gemini"):
        raise ValueError("Identity registry cannot claim OpenAI or Google Gemini affiliation")
    if indep.get("is_chatgpt_clone") or indep.get("is_gemini_clone") or indep.get("uses_secret_external_llm"):
        raise ValueError("Identity registry cannot claim ChatGPT/Gemini cloning or secret external LLM use")

    owner = record["ownership_and_development"]
    if owner.get("official_kaggle_url") != "https://www.kaggle.com/ypy0001":
        raise ValueError("Official Kaggle URL mismatch in identity registry")
    if owner.get("official_repository_url") != "https://github.com/ypy999/NourAI":
        raise ValueError("Official repository URL mismatch in identity registry")

    for cap in record.get("verified_capabilities", []):
        if cap.get("test_passed") is not True:
            raise ValueError(f"Verified capability '{cap.get('id')}' cannot be listed without test_passed=True")
        if not str(cap.get("source", "")).strip():
            raise ValueError(f"Verified capability '{cap.get('id')}' missing verification source")


def load_identity_registry(path: Path = DEFAULT_IDENTITY_PATH) -> Dict[str, Any]:
    """Load, validate, and return the single authoritative Jebrex AI identity knowledge record."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Central identity registry not found: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    validate_identity_record(data)
    data["_meta"] = {
        "path": str(target).replace("\\", "/"),
        "sha256": _sha256_file(target),
    }
    return data


def _bump_minor_version(ver: str) -> str:
    parts = [int(p) for p in ver.strip().split(".")]
    while len(parts) < 3:
        parts.append(0)
    return f"{parts[0]}.{parts[1] + 1}.0"


class IdentityVersionManager:
    """Versioned self-knowledge governance engine (Phase I Section 8).

    Rules enforced strictly:
    - Never announces a new capability unless `test_passed=True` and `source` evidence is provided.
    - Preserves previous version snapshot before writing any update.
    - Re-validates consistency invariants automatically after every update.
    - Blocks user conversation data ingestion unless explicit consent + PII scrubbing are verified.
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_IDENTITY_PATH,
        history_dir: Optional[Path] = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.history_dir = Path(history_dir) if history_dir is not None else (self.registry_path.parent / "history")

    def current(self) -> Dict[str, Any]:
        return load_identity_registry(self.registry_path)

    def register_verified_capability(
        self,
        *,
        capability_id: str,
        arabic: str,
        english: str,
        french: str,
        test_passed: bool,
        evidence_source: str,
        change_summary: str,
        new_version: Optional[str] = None,
        training_source_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a newly verified capability ONLY after its tests pass, preserving the previous version."""
        if test_passed is not True:
            raise ValueError(
                f"Cannot announce capability '{capability_id}' before passing verification tests (test_passed must be True)."
            )
        if not str(evidence_source).strip():
            raise ValueError("evidence_source is mandatory to register a verified capability.")
        if training_source_meta is not None:
            from training.continuous_improvement import validate_training_data_consent
            validate_training_data_consent(training_source_meta)

        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        validate_identity_record(raw)

        prev_version = str(raw["identity_version"])
        self.history_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.history_dir / f"jebrex_identity_v{prev_version}.json"
        if not archive_path.exists():
            shutil.copy2(self.registry_path, archive_path)

        updated = copy.deepcopy(raw)
        next_version = new_version or _bump_minor_version(prev_version)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        caps: List[Dict[str, Any]] = list(updated.get("verified_capabilities", []))
        caps = [c for c in caps if c.get("id") != capability_id]
        caps.append(
            {
                "id": capability_id,
                "arabic": arabic.strip(),
                "english": english.strip(),
                "french": french.strip(),
                "test_passed": True,
                "source": evidence_source.strip(),
                "verified_date": today,
            }
        )
        updated["verified_capabilities"] = caps
        updated["identity_version"] = next_version
        updated["verified_at_utc"] = now_utc
        updated["version_history"] = list(updated.get("version_history", [])) + [
            {
                "version": next_version,
                "date": today,
                "summary": change_summary.strip(),
                "source": evidence_source.strip(),
            }
        ]

        validate_identity_record(updated)
        self.registry_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {
            "previous_version": prev_version,
            "archived_snapshot": str(archive_path).replace("\\", "/"),
            "new_version": next_version,
            "verified_sha256": _sha256_file(self.registry_path),
            "consistency_revalidated": True,
        }
