import json
from pathlib import Path

from corpus.leakage_audit import audit_split_leakage_and_similarity


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_leakage_auditor_detects_injected_document_and_translation_group_leakage(tmp_path):
    _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "text": "English text in train split for document 101.",
                "source_id": "wto_public_corpus_en_fr",
                "document_id": "101",
                "translation_group_id": "101:1",
                "source_language_code": "eng",
                "split_group_id": "g1",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "validation.jsonl",
        [
            {
                "text": "Texte français dans validation pour le même document 101.",
                "source_id": "wto_public_corpus_en_fr",
                "document_id": "101",
                "translation_group_id": "101:1",
                "source_language_code": "fra",
                "split_group_id": "g2",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "text": "Distinct text in test split for document 202.",
                "source_id": "wto_public_corpus_en_fr",
                "document_id": "202",
                "translation_group_id": "202:1",
                "source_language_code": "eng",
                "split_group_id": "g3",
            }
        ],
    )
    audit = audit_split_leakage_and_similarity(tmp_path)
    assert audit["status"] == "FAIL"
    assert audit["leakage_counts"]["document_id_leakage"] == 1
    assert audit["leakage_counts"]["translation_group_leakage"] == 1


def test_phase10_3_splits_have_zero_document_and_translation_group_leakage():
    root = Path(__file__).resolve().parents[1]
    audit = audit_split_leakage_and_similarity(root / "data/processed/phase10_3/splits")
    assert audit["status"] == "PASS", audit["leakage_counts"]
    assert all(audit["acceptance_criteria"].values())
    assert audit["unique_documents_with_explicit_id"] == 134  # 58 QNL + 76 WTO paired documents with seed=1337
    assert audit["unique_translation_groups"] > 20000
