import pytest

from data.license_manager import LicenseManager
from data.license_policy import check_corpus_licenses, require_compatible
from data.source_registry import SourceRegistry


def test_license_history_and_manual_review_are_auditable(tmp_path):
    manager=LicenseManager(tmp_path/"metadata.sqlite3")
    try:
        manager.seed_catalog()
        with pytest.raises(PermissionError):
            manager.ensure_approved("tatoeba_general")
        manager.review("tatoeba_general","VERIFIED",reviewer="reviewer-1",rationale="terms and attribution reviewed",
                       evidence_url="https://tatoeba.org/en/terms_of_use",license_snapshot="CC-BY-2.0-FR",
                       license_id="CC-BY-2.0-FR",approve_for_ingestion=True)
        assert manager.ensure_approved("tatoeba_general")["state"]=="VERIFIED"
        history=manager.history("tatoeba_general")
        assert len(history)==2 and history[-1]["reviewer"]=="reviewer-1"
    finally:
        manager.close()


def test_unknown_and_mixed_license_sources_are_blocked():
    registry=SourceRegistry()
    cc0=registry.get("tatoeba_cc0")
    unknown=registry.get("oscar")
    result=check_corpus_licenses([cc0,unknown])
    assert not result["compatible"]
    with pytest.raises(PermissionError):
        require_compatible([cc0,unknown])
    assert require_compatible([cc0])["compatible"]
