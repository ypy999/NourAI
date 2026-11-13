from data.source_registry import SourceRegistry
from corpus.pipeline import load_phase8_config


def test_source_registry_records_provenance_license_and_scope():
    registry=SourceRegistry()
    cc0=registry.get("tatoeba_cc0")
    assert cc0["status"]=="VERIFIED" and cc0["approved_for_ingestion"] is True
    assert cc0["license_url"] and cc0["source_license_url"] and cc0["checked_at"]
    assert cc0["approximate_size_bytes"]==7963537
    assert registry.get("tatoeba_general")["status"]=="NEEDS_REVIEW"
    assert registry.get("wikimedia")["sharealike_required"] is True
    assert registry.get("oscar")["status"]=="UNKNOWN_LICENSE"


def test_phase8_config_declares_supported_formats_and_per_language_settings():
    config=load_phase8_config()
    assert {"jsonl","json","txt","csv","parquet","zip"}.issubset(set(config["supported_formats"]))
    assert set(config["language_settings"])=={"ara","eng","fra"}
    assert config["license_policy"]["unknown_license_allowed"] is False
