import json
from pathlib import Path
import shutil
import sqlite3

from configs.configuration import load_config
from data.license_manager import LicenseManager
from data.source_registry import SourceRegistry
from data.splitting import split_sources
from data.storage_manager import CorpusMetadataStore
from tokenizer.byte_tokenizer import ByteTokenizer


def test_sqlite_schema_migration_and_catalog_sync(tmp_path):
    db_path = tmp_path / "data/metadata/legacy.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE records(
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                source_record_id TEXT,
                raw_sha256 TEXT,
                language TEXT,
                language_confidence REAL,
                language_method TEXT,
                quality_score REAL,
                license_id TEXT NOT NULL,
                record_status TEXT NOT NULL,
                split TEXT,
                metadata_json TEXT NOT NULL
            );
            """
        )

    store = CorpusMetadataStore(tmp_path, "data/metadata/legacy.sqlite3")
    try:
        cols = {row[1] for row in store.connection.execute("PRAGMA table_info(records)").fetchall()}
        for required_col in ("document_id", "translation_group_id", "domain", "ingestion_timestamp", "content_hash"):
            assert required_col in cols

        synced_count = store.sync_catalog()
        expected_count = len(SourceRegistry().list_sources())
        assert synced_count == expected_count == 14
        assert store.status()["sources"] == 14

        lm = LicenseManager(db_path)
        try:
            current = {item["source_id"]: item for item in lm.list_current()}
            assert len(current) == 14
            for approved_id in ("tatoeba_cc0", "qnl_arabic_ocr_v2", "wto_public_corpus_en_fr"):
                assert current[approved_id]["state"] == "VERIFIED"
                assert current[approved_id]["approved_for_ingestion"] is True
        finally:
            lm.close()
    finally:
        store.close()


def test_provenance_indexing_for_tatoeba_qnl_and_wto(tmp_path):
    store = CorpusMetadataStore(tmp_path, "data/metadata/phase10_3_test.sqlite3")
    try:
        store.sync_catalog()
        store.register_dataset(
            "multilingual:phase10_3",
            "NourAI Multilingual Corpus (Phase 10.3)",
            "v1",
            "tatoeba_cc0",
            "CC0-1.0; PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA; WTO_PUBLIC_CORPUS_TERMS",
            status="CONDITIONALLY_READY",
        )
        rows = [
            {
                "text": "This is a Tatoeba sentence.",
                "raw_text_sha256": "a" * 64,
                "content_hash": "a" * 64,
                "source_id": "tatoeba_cc0",
                "source_record_id": "101",
                "source_language_code": "eng",
                "license_id": "CC0-1.0",
                "ingestion_timestamp": "2026-11-13T20:00:00+00:00",
            },
            {
                "text": "كتاب عربي من مكتبة قطر الوطنية.",
                "raw_text_sha256": "b" * 64,
                "content_hash": "b" * 64,
                "source_id": "qnl_arabic_ocr_v2",
                "source_record_id": "i10027154",
                "document_id": "i10027154",
                "source_language_code": "ara",
                "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA",
                "ingestion_timestamp": "2026-11-13T20:00:00+00:00",
                "adapter_version": "qnl_arabic_ocr_v2_adapter_v1",
            },
            {
                "text": "WTO trade policy review paragraph.",
                "raw_text_sha256": "c" * 64,
                "content_hash": "c" * 64,
                "source_id": "wto_public_corpus_en_fr",
                "source_record_id": "600001:eng",
                "document_id": "10863843",
                "translation_group_id": "10863843:1",
                "domain": "TPR",
                "source_language_code": "eng",
                "license_id": "WTO_PUBLIC_CORPUS_TERMS",
                "ingestion_timestamp": "2026-11-13T20:00:00+00:00",
                "adapter_version": "wto_en_fr_adapter_v1",
            },
            {
                "text": "Paragraphe d'examen des politiques commerciales de l'OMC.",
                "raw_text_sha256": "d" * 64,
                "content_hash": "d" * 64,
                "source_id": "wto_public_corpus_en_fr",
                "source_record_id": "600001:fra",
                "document_id": "10863843",
                "translation_group_id": "10863843:1",
                "domain": "TPR",
                "source_language_code": "fra",
                "license_id": "WTO_PUBLIC_CORPUS_TERMS",
                "ingestion_timestamp": "2026-11-13T20:00:00+00:00",
                "adapter_version": "wto_en_fr_adapter_v1",
            },
        ]
        split_file = tmp_path / "data/processed/phase10_3/splits/train.jsonl"
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

        indexed = store.index_jsonl_metadata("multilingual:phase10_3", split_file, split="train")
        assert indexed == 4

        summary = store.provenance_summary("multilingual:phase10_3")
        assert summary["records_indexed"] == 4
        assert summary["with_source_record_id"] == 4
        assert summary["with_document_id"] == 3
        assert summary["with_translation_group_id"] == 2
        assert summary["with_domain"] == 2
        assert summary["with_ingestion_timestamp"] == 4
        assert summary["with_content_hash"] == 4
        assert summary["by_split"] == {"train": 4}
        assert summary["by_domain"]["TPR"] == 2
    finally:
        store.close()


def test_compact_normalized_preserves_raw_sha256_through_split(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / "configs/default.json", tmp_path / "configs/default.json")

    norm_file = tmp_path / "normalized.jsonl"
    original_raw_sha = "1" * 64
    original_norm_sha = "2" * 64
    cleaned_sha = "3" * 64
    rows = [
        {
            "text": f"Cleaned text for document {i} with enough characters.",
            "source_id": "qnl_arabic_ocr_v2",
            "source_record_id": f"i100{i}",
            "document_id": f"i100{i}",
            "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA",
            "raw_text_sha256": original_raw_sha,
            "normalized_text_sha256": original_norm_sha,
            "cleaned_text_sha256": cleaned_sha,
        }
        for i in range(6)
    ]
    norm_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    splits_dir = tmp_path / "splits"
    project_config = load_config(tmp_path / "configs/default.json", ByteTokenizer())
    split_sources(
        [norm_file],
        splits_dir,
        config=project_config,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=1337,
    )

    train_records = [
        json.loads(line)
        for line in (splits_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_records) > 0
    for rec in train_records:
        assert rec["raw_text_sha256"] == original_raw_sha
        assert rec["normalized_text_sha256"] == original_norm_sha
        assert "raw_text" not in rec
        assert "normalized_text" not in rec
        assert "cleaned_text" not in rec
