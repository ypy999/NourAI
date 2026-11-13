import json
import sqlite3
from pathlib import Path

from data.content_quality import evaluate_content_record
from data.phase10_readiness import assess_phase10_readiness, required_sample_size
from data.registry_sync import RegistrySync


def test_content_quality_is_layered_and_semantic_is_not_claimed():
    result = evaluate_content_record("مرحبا, hello 😊")
    assert result["structural_quality"]["utf8_valid"]
    assert result["linguistic_quality"]["mixed_language_or_script"]
    assert result["semantic_quality"]["status"] == "NOT_EVALUATED"
    assert result["combined_score"] is None


def test_readiness_uses_configurable_sample_plan_and_fails_closed():
    assert required_sample_size(10000, confidence_level=.95, margin_of_error=.05) == 370
    result = assess_phase10_readiness(language_statistics={}, split_language_statistics={}, language_diversity={},
        used_sources=[], split_validation={}, quality_statistics={}, tokenizer_compatibility={})
    assert result["overall_status"] == "UNKNOWN"
    assert all(status == "UNKNOWN" for status in result["language_statuses"].values())


def test_registry_sync_dry_run_is_non_mutating(tmp_path):
    config = tmp_path / "configs"; config.mkdir()
    (config / "dataset_sources.json").write_text(json.dumps({"new": {"name": "New", "license_id": "CC0-1.0", "status": "NEEDS_REVIEW"}}), encoding="utf-8")
    data = tmp_path / "data" / "metadata"; data.mkdir(parents=True)
    db_path = data / "corpus.sqlite3"
    con = sqlite3.connect(db_path)
    con.executescript("CREATE TABLE license_current(source_id TEXT PRIMARY KEY,state TEXT,license_id TEXT,payload_json TEXT,updated_at TEXT); CREATE TABLE license_history(event_id INTEGER PRIMARY KEY AUTOINCREMENT,source_id TEXT,previous_state TEXT,new_state TEXT,reviewer TEXT,rationale TEXT,evidence_url TEXT,license_snapshot TEXT,created_at TEXT); CREATE TABLE operations(operation_id INTEGER PRIMARY KEY AUTOINCREMENT,operation TEXT,status TEXT,source_id TEXT,details_json TEXT,created_at TEXT);")
    con.commit(); con.close()
    sync = RegistrySync(tmp_path)
    before = sqlite3.connect(db_path).execute("select count(*) from license_current").fetchone()[0]
    result = sync.sync(dry_run=True)
    after = sqlite3.connect(db_path).execute("select count(*) from license_current").fetchone()[0]
    assert result["status"] == "DRY_RUN" and result["comparison"]["missing_in_sqlite"] == ["new"]
    assert before == after == 0


def test_registry_sync_rejects_paths_outside_project(tmp_path):
    import pytest

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.sqlite3"
    with pytest.raises(ValueError, match="database path must remain under the project root"):
        RegistrySync(root, database_path=outside)
    with pytest.raises(ValueError, match="source registry path must remain under the project root"):
        RegistrySync(root, registry_path="../outside.json")
