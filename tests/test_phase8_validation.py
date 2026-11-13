import json

from corpus.validation import validate_jsonl
from data.raw_storage import make_raw_record


def test_validation_reports_duplicates_suspicious_text_and_lengths(tmp_path):
    path=tmp_path/"records.jsonl"
    rows=[]
    for text in ("Hello world", " hello   world ", "x"*45):
        row=make_raw_record(text,source_id="tatoeba_cc0",license_id="CC0-1.0"); row["text"]=text
        rows.append(row)
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    result=validate_jsonl(path,require_provenance=True)
    assert result["duplicate_records"]==1 and result["suspicious_records"]==1
    assert result["length_characters"]["max"]==45 and not result["valid"]


def test_validation_blocks_unknown_license_and_invalid_utf8(tmp_path):
    path=tmp_path/"bad.jsonl"
    path.write_bytes(b'{"text":"hello","source_id":"oscar","license_id":"UNKNOWN"}\n\xff\n')
    result=validate_jsonl(path,require_provenance=True)
    assert not result["valid"] and result["unknown_source_records"]==1 and result["unknown_license_records"]==1
