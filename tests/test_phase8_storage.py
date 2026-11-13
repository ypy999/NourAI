import json
import pytest

from data.storage_manager import CorpusMetadataStore


def test_sqlite_metadata_catalog_indexes_sources_files_records_and_restores(tmp_path):
    store=CorpusMetadataStore(tmp_path)
    try:
        source={"source_id":"tatoeba_cc0","name":"Tatoeba CC0","url":"https://example.invalid","license_id":"CC0-1.0","status":"VERIFIED"}
        store.register_source(source)
        store.register_dataset("ds1","Tatoeba","v1","tatoeba_cc0","CC0-1.0",status="IMPORTED")
        path=tmp_path/"data/import.jsonl"; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"text":"hello","source_id":"tatoeba_cc0","license_id":"CC0-1.0","source_record_id":"1"})+"\n",encoding="utf-8")
        file_meta=store.register_file("ds1",path,stage="IMPORTED_RAW_TEXT",record_count=1)
        assert len(file_meta["sha256"])==64
        assert store.index_jsonl_metadata("ds1",path)==1
        assert store.index_jsonl_metadata("ds1",path)==1
        assert store.status()["records_indexed"]==1
        backup=store.backup_to("data/metadata-backup.sqlite3")
    finally:
        store.close()
    restored=tmp_path/"restored.sqlite3"
    CorpusMetadataStore.restore_from(backup,restored)
    import sqlite3
    with sqlite3.connect(restored) as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0]==1
    with pytest.raises(FileExistsError):
        CorpusMetadataStore.restore_from(backup,restored)


def test_sqlite_store_rejects_paths_outside_project(tmp_path):
    store=CorpusMetadataStore(tmp_path)
    try:
        with pytest.raises(ValueError):
            store.resolve("../outside.sqlite3")
    finally:
        store.close()


def test_sqlite_metadata_does_not_fabricate_source_record_ids(tmp_path):
    store=CorpusMetadataStore(tmp_path)
    try:
        store.register_source({"source_id":"tatoeba_cc0","name":"Tatoeba CC0","url":"https://example.invalid","license_id":"CC0-1.0","status":"VERIFIED"})
        store.register_dataset("ds1","Tatoeba","v1","tatoeba_cc0","CC0-1.0",status="IMPORTED")
        path=tmp_path/"data/split.jsonl"; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"text":"hello","source_id":"tatoeba_cc0","license_id":"CC0-1.0"})+"\n",encoding="utf-8")

        assert store.index_jsonl_metadata("ds1",path,split="train")==1
        row=store.connection.execute("SELECT source_record_id,split,metadata_json FROM records").fetchone()
        assert row[0] is None
        assert row[1]=="train"
        assert json.loads(row[2])["record_number"]==1
    finally:
        store.close()
