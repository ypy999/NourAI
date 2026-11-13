import json
import pytest

from corpus.validation import validate_splits
from data.splitting import split_records


def test_document_groups_never_cross_splits_and_languages_are_stratified(tmp_path):
    records = []
    for language, prefix in (("en", "The science article"), ("ar", "هذا مقال عربي"), ("fr", "La science article")):
        for document in range(6):
            for segment in range(4):
                records.append({"text": f"{prefix} {document} segment {segment} has distinct text.",
                                "source_id": "fixture", "document_id": f"{language}-doc-{document}",
                                "record_id": f"{language}-{document}-{segment}"})
    output = tmp_path / "split"
    result = split_records(records, output, seed=18, checkpoint_batch_size=8)
    assert result.group_count == 18
    assert result.records_without_document_id == 0
    split_for_document = {}
    for split in ("train", "validation", "test"):
        for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            split_for_document.setdefault(row["document_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in split_for_document.values())
    validation = validate_splits({name: output / f"{name}.jsonl" for name in ("train", "validation", "test")})
    assert validation["valid"]
    assert validation["document_group_leakage"] == 0
    # Every sufficiently represented source-language stratum is distributed.
    assert result.split_counts["train"] > result.split_counts["validation"] > 0
    assert result.split_counts["train"] > result.split_counts["test"] > 0


def test_missing_document_ids_are_independent_groups(tmp_path):
    records = [{"text": f"Independent record number {index} has enough distinct words."} for index in range(12)]
    output = tmp_path / "split"
    result = split_records(records, output, seed=3)
    assert result.group_count == 12
    assert result.records_without_document_id == 12
    assert result.split_counts == {"train": 10, "validation": 1, "test": 1}


def test_split_uses_precomputed_language_stratum_for_large_documents(tmp_path, monkeypatch):
    import data.splitting as splitting

    def classifier_must_not_run(_text):
        pytest.fail("precomputed split_language should avoid rescanning a large document")

    monkeypatch.setattr(splitting, "classify_language", classifier_must_not_run)
    records = [
        {"text": f"large text payload {language} {index}", "split_language": language,
         "document_id": f"{language}-{index}", "source_id": "qnl"}
        for language in ("arabic", "english", "french")
        for index in range(3)
    ]
    result = split_records(records, tmp_path / "preclassified", seed=29)
    assert result.unique_records == 9
    assert set(result.language_counts["train"]) == {"arabic", "english", "french"}


def test_validator_detects_document_group_leakage_even_for_different_text(tmp_path):
    paths = {}
    for split, text in (("train", "first document fragment text"),
                        ("validation", "second fragment from same document"),
                        ("test", "an independent test document")):
        path = tmp_path / f"{split}.jsonl"
        group = "same-document" if split != "test" else "test-document"
        path.write_text(json.dumps({"text": text, "split_group_id": group}) + "\n", encoding="utf-8")
        paths[split] = path
    result = validate_splits(paths)
    assert result["document_group_leakage"] == 1
    assert not result["valid"]
