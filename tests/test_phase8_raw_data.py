import json

from data.raw_storage import RawCorpusStore, make_raw_record, verify_raw_record


def test_raw_storage_preserves_original_text_and_checksum(tmp_path):
    text="  إِنَّها 👩‍👩‍👧‍👦  "
    row=make_raw_record(text,source_id="tatoeba_cc0",dataset_version="snapshot-x",license_id="CC0-1.0",source_record_id="123")
    assert row["raw_text"]==text and row["raw_stage"]=="preserved_unchanged"
    assert verify_raw_record(row)
    row["raw_text"]="tampered"
    assert not verify_raw_record(row)


def test_raw_store_streams_jsonl_without_modifying_rows(tmp_path):
    store=RawCorpusStore(tmp_path)
    rows=[make_raw_record("مرحبا",source_id="tatoeba_cc0",license_id="CC0-1.0")]
    path=store.write_records("data/raw/test.jsonl",rows)
    restored=list(store.iter_records(path))
    assert restored==rows and json.loads(path.read_text(encoding="utf-8"))["raw_text"]=="مرحبا"
