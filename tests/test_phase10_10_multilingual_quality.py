import json
from pathlib import Path

from corpus.multilingual_quality import validate_multilingual_quality


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_multilingual_quality_detects_missing_language_and_replacement_chars(tmp_path):
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"text": "نص عربي سليم في التدريب.", "source_language_code": "ara"},
            {"text": "Valid English text in train.", "source_language_code": "eng"},
            {"text": "Texte français avec caractère corrompu \ufffd.", "source_language_code": "fra"},
        ],
    )
    _write_jsonl(
        tmp_path / "validation.jsonl",
        [
            {"text": "Valid English text only.", "source_language_code": "eng"},
        ],
    )
    _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {"text": "Valid English text only.", "source_language_code": "eng"},
        ],
    )
    res = validate_multilingual_quality(tmp_path)
    assert res["status"] == "FAIL"
    assert res["acceptance_criteria"]["all_splits_cover_arabic_english_french"] is False
    assert res["acceptance_criteria"]["zero_utf8_or_nfc_encoding_errors"] is False
    assert res["encoding_integrity"]["replacement_char_records"] == 1


def test_phase10_3_multilingual_quality_passes():
    root = Path(__file__).resolve().parents[1]
    res = validate_multilingual_quality(root / "data/processed/phase10_3/splits")
    assert res["status"] == "PASS", res
    assert all(res["acceptance_criteria"].values())
    for split_name in ("train", "validation", "test"):
        for lang in ("arabic", "english", "french"):
            assert res["split_language_record_counts"][split_name][lang] > 0
