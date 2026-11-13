import json

from configs.configuration import ProjectConfig
from data.processor import DataPipeline, DataPipelineStats
from data.provenance import canonical_record
from data.splitting import split_records
from datasets.dataset import TokenSequenceDataset


def test_canonical_record_preserves_source_and_distinguishes_missing_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {"raw_text": "مرحبا بالعالم hello", "text": "مرحبا بالعالم hello", "source_id": "tatoeba_cc0",
           "license_id": "CC0-1.0", "source_record_id": "r-1", "source_language_code": "ara",
           "document_id": "doc-7", "imported_at": "2026-01-01T00:00:00Z"}
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    records = list(DataPipeline()._iter_records([source], DataPipelineStats()))
    actual = records[0]
    assert actual["raw_text"] == row["raw_text"]
    assert actual["source_id"] == "tatoeba_cc0"
    assert actual["source_name"] == "Tatoeba sentences CC0 export"
    assert actual["license_status"] == "VERIFIED"
    assert actual["document_id"] == "doc-7"
    assert actual["original_language"] == "ara"
    assert actual["detected_language"] == "mixed"
    assert actual["source_file"] == str(source.resolve())
    assert actual["source_line"] == 1
    assert len(actual["record_id"]) == 64
    assert actual["ingestion_timestamp"] == row["imported_at"]
    assert "document_id" not in actual["provenance_missing"]


def test_legacy_records_get_stable_ids_without_fabricated_source_fields(tmp_path):
    raw = {"text": "Legacy text only"}
    first = canonical_record(raw, source_file=tmp_path / "old.jsonl", source_line=4)
    second = canonical_record(raw, source_file=tmp_path / "old.jsonl", source_line=4)
    assert first["record_id"] == second["record_id"]
    assert first["source_id"] is None
    assert first["license_id"] is None
    assert first["document_id"] is None
    assert "license_id" in first["provenance_missing"]


def test_provenance_survives_deduplication_splitting_and_dataset(tmp_path):
    rows = [{"text": f"English source sentence number {i} with enough content.",
             "raw_text": f"English source sentence number {i} with enough content.",
             "source_id": "tatoeba_cc0", "license_id": "CC0-1.0", "license_status": "VERIFIED",
             "document_id": f"doc-{i}", "record_id": f"record-{i}", "content_hash": f"hash-{i}"}
            for i in range(12)]
    output = tmp_path / "splits"
    result = split_records(rows, output, seed=77, checkpoint_batch_size=4)
    assert result.unique_records == 12
    split_rows = []
    for split in ("train", "validation", "test"):
        split_rows.extend(json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    assert all(row["source_id"] == "tatoeba_cc0" and row["license_status"] == "VERIFIED" for row in split_rows)
    assert all(row.get("document_id") for row in split_rows)

    train_path = output / "train.jsonl"
    dataset = TokenSequenceDataset(train_path, config=ProjectConfig(context_length=16))
    try:
        metadata = dataset.metadata_for_sequence(0)
        assert metadata
        assert all(item["source_id"] == "tatoeba_cc0" for item in metadata)
        assert all(item["document_id"] for item in metadata)
    finally:
        dataset._temp.cleanup()
