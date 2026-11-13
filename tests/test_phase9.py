import json

import pytest

from data.domain_analysis import analyze_arabic_text, classify_domains
from data.language_balancing import analyze_language_balance, validate_target_ratios
from data.phase9_readiness import assess_phase9_readiness
from data.phase9_pipeline import analyze_corpus, write_phase9_reports
from data.source_registry import SourceRegistry


def test_domain_analysis_marks_heuristics_and_metadata():
    heuristic = classify_domains("The science experiment studies energy.")
    assert heuristic["method"] == "multilingual_lexicon_heuristic"
    assert "Science" in heuristic["domains"]
    metadata = classify_domains("texte", source_domains="Science")
    assert metadata["method"] == "source_metadata"
    assert metadata["status"] == "source_label_not_independently_verified"


def test_arabic_diagnostics_do_not_change_text():
    text = "مَرْحَبًا بالعالم 😊؟"
    result = analyze_arabic_text(text)
    assert result["arabic_letter_count"] > 0
    assert result["arabic_diacritic_count"] > 0
    assert result["arabic_punctuation_count"] == 1
    assert result["raw_text_modified"] is False


def test_balance_validation_and_empty_group_weights():
    with pytest.raises(ValueError):
        validate_target_ratios({"arabic": 1})
    rows = [{"text": "مرحبا", "language": "arabic", "quality_score": 90, "source_id": "a"},
            {"text": "Bonjour merci", "language": "french", "quality_score": 80, "source_id": "b"}]
    result = analyze_language_balance(rows)
    assert result["records"] == 2
    assert result["languages"]["arabic"]["tokens_including_eos"] == 11
    assert result["sampling_weights"]["english"] is None
    assert result["records_modified_or_deleted"] == 0


def test_readiness_separates_training_scope_and_missing_coverage():
    result = assess_phase9_readiness(language_statistics={}, split_language_statistics={}, used_sources=[],
                                     split_validation={"valid": True, "duplicate_leakage": 0,
                                                       "duplicates_within_splits": 0},
                                     balance_statistics={"languages": {}}, training_allowed=False)
    assert result["status"] == "NOT_READY"
    assert result["training_readiness"]["training_started"] is False
    assert result["linguistic_readiness"]["status"] == "NOT_READY"


def test_streamed_corpus_analysis_validates_splits_and_reports(tmp_path):
    imported = tmp_path / "imported.jsonl"
    imported.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
        {"text": "Hello, world!", "raw_text": "Hello, world!", "source_language_code": "eng", "source_id": "sample", "license_id": "CC0-1.0"},
        {"text": "Bonjour le monde!", "raw_text": "Bonjour le monde!", "source_language_code": "fra", "source_id": "sample", "license_id": "CC0-1.0"},
        {"text": "مرحبا بالعالم!", "raw_text": "مرحبا بالعالم!", "source_language_code": "ara", "source_id": "sample", "license_id": "CC0-1.0"},
    ]) + "\n", encoding="utf-8")
    # Split validator requires three nonempty splits; split texts must not overlap.
    split_paths = {}
    for name, text in (("train", "Hello, world!"), ("validation", "Bonjour le monde!"), ("test", "مرحبا بالعالم!")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"text": text}, ensure_ascii=False) + "\n", encoding="utf-8")
        split_paths[name] = path

    class Registry:
        def list_sources(self):
            return [{"source_id": "sample", "status": "VERIFIED", "approved_for_ingestion": True,
                     "license_id": "CC0-1.0"}]

    report = analyze_corpus(imported, split_paths, registry=Registry())
    assert report["records"] == 3
    assert report["split_validation"]["valid"]
    assert report["split_hash_leakage"] == 0
    assert report["languages"]["arabic"]["records"] == 1
    paths = write_phase9_reports(report, tmp_path / "reports", project_root=tmp_path)
    assert all(path.exists() for path in paths)
    assert json.loads((tmp_path / "reports/phase9_corpus_report.json").read_text(encoding="utf-8"))["training_started"] is False


def test_candidate_sources_are_not_approved_and_existing_cc0_stays_usable():
    registry = SourceRegistry()
    assert registry.verified_for_ingestion("tatoeba_cc0") is True
    assert registry.verified_for_ingestion("common_voice_ar_27") is False
    assert registry.get("hplt_v1_2")["license_id"] == "UNKNOWN_LICENSE"

