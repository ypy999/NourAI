import json
from pathlib import Path

from corpus.pipeline import load_phase8_config
from corpus.reproducibility import audit_artifacts, check_split_decisions, verify_run_manifest
from data.storage_manager import CorpusMetadataStore


def test_phase10_3_regenerated_splits_and_run_manifest():
    root = Path(__file__).resolve().parents[1]
    config = load_phase8_config(root / "configs/phase10_3_corpus.json")

    run_manifest_path = root / "data/processed/phase10_3/run_manifest.json"
    assert run_manifest_path.is_file(), "run_manifest.json must exist in data/processed/phase10_3/"

    verification = verify_run_manifest(root, run_manifest_path)
    assert verification["valid"] is True, verification["errors"]

    manifest_data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    split_decisions = manifest_data["split_decisions"]
    findings = check_split_decisions(split_decisions, config)
    assert findings == [], f"Unexpected split decision findings: {findings}"
    assert split_decisions["requested_ratios"] == {"train": 0.9, "validation": 0.05, "test": 0.05}
    assert split_decisions["seed"] == 1337
    assert split_decisions["grouping_field"] == "document_id"

    audit = audit_artifacts(root, targets=["data/processed/phase10_3"])
    for item in audit["artifacts"]:
        if item["kind"] in {"imported", "normalized", "split"}:
            assert item["reproducible"] == "YES", f"Artifact not reproducible: {item}"
    assert audit["summary"]["by_reproducibility"].get("NO", 0) == 0
    assert audit["summary"]["by_reproducibility"].get("PARTIAL", 0) == 0


def test_phase10_3_sqlite_provenance_completeness():
    root = Path(__file__).resolve().parents[1]
    store = CorpusMetadataStore(root, "data/metadata/phase10_3_corpus.sqlite3")
    try:
        status = store.status()
        assert status["sources"] == 14
        summary = store.provenance_summary("multilingual:phase10_3")
        assert summary["records_indexed"] > 50000
        assert summary["with_source_record_id"] == summary["records_indexed"]
        assert summary["with_ingestion_timestamp"] == summary["records_indexed"]
        assert summary["with_content_hash"] == summary["records_indexed"]
        assert summary["with_document_id"] > 0
        assert summary["with_translation_group_id"] > 0
        assert set(summary["by_split"].keys()) == {"train", "validation", "test"}
    finally:
        store.close()
