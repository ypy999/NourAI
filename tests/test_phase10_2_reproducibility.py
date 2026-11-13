import hashlib
import json
from pathlib import Path

import pytest

from corpus import reproducibility as repro
from corpus.acquisition import sha256_file
from corpus.pipeline import load_phase8_config
from data.splitting import split_records


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sidecar_for(root, relative, source_id, url, license_id):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"fixture bytes for " + source_id.encode())
    meta = {
        "source_id": source_id,
        "source_url": url,
        "license_id": license_id,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_audit_marks_adhoc_enrichment_and_dropped_transformations(tmp_path):
    root = tmp_path
    _write_jsonl(
        root / "data/processed/phase10_2/qnl/imported.jsonl",
        [{"text": "نص", "source_id": "qnl_arabic_ocr_v2", "qnl_metadata": {"title": "x"},
          "arabic_script_ratio": 0.9, "provenance_id_source": "QNL", "document_id_status": "SOURCE_PROVIDED"}],
    )
    _write_jsonl(
        root / "data/processed/phase10_2/qnl/normalized.jsonl",
        [{"text": "نص", "source_id": "qnl_arabic_ocr_v2", "record_id": "r1",
          "normalized_text_sha256": "a", "cleaned_text_sha256": "b"}],
    )
    _write_jsonl(
        root / "data/processed/phase10_2/splits_v1/train.jsonl",
        [{"text": "نص", "split_group_id": "g1", "source_id": "qnl_arabic_ocr_v2"}],
    )
    partial_path = root / "data/processed/phase10_2/wto/imported.jsonl.partial"
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("partial\n", encoding="utf-8")
    archive = root / "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"PK fixture")

    audit = repro.audit_artifacts(root)
    by_path = {item["path"]: item for item in audit["artifacts"]}

    imported = by_path["data/processed/phase10_2/qnl/imported.jsonl"]
    assert imported["reproducible"] == "NO"
    assert "qnl_metadata" in imported["reason"]
    assert imported["sha256"] == hashlib.sha256((root / imported["path"]).read_bytes()).hexdigest()

    normalized = by_path["data/processed/phase10_2/qnl/normalized.jsonl"]
    assert normalized["reproducible"] == "NO"
    assert "raw_text" in normalized["reason"]

    split = by_path["data/processed/phase10_2/splits_v1/train.jsonl"]
    assert split["reproducible"] == "PARTIAL"

    partial = by_path["data/processed/phase10_2/wto/imported.jsonl.partial"]
    assert partial["reproducible"] == "NO" and partial["kind"] == "incomplete_attempt"

    raw = by_path["data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip"]
    assert raw["reproducible"] == "YES" and raw["kind"] == "raw_source"


def test_recorded_phase10_2_split_decisions_conflict_with_approved_config():
    report = {
        "requested_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "seed": 10202,
        "grouping_field": "document_id (QNL QNL RECORD #(ITEM); WTO official IdxID; Tatoeba source record)",
    }
    findings = repro.check_split_decisions(report, load_phase8_config())
    fields = {item["field"] for item in findings}
    assert fields == {"split_ratios.train", "split_ratios.validation", "split_ratios.test", "random_seed"}
    assert all(item["severity"] == "BLOCKER" for item in findings)


def test_conforming_split_decisions_have_no_findings():
    config = load_phase8_config()
    report = {"requested_ratios": config["split_ratios"], "seed": config["random_seed"],
              "grouping_field": "document_id"}
    assert repro.check_split_decisions(report, config) == []


def test_plan_accepts_supported_source_with_verified_sidecar(tmp_path):
    path = _sidecar_for(tmp_path, "data/raw/external/tatoeba_cc0.tar.bz2", "tatoeba_cc0",
                        "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2", "CC0-1.0")
    plan = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3",
                            source_ids=["tatoeba_cc0"], verify_hashes=True)
    assert plan["ready"] is True, plan["blockers"]
    assert plan["sources"][0]["support"]["support"] == "SUPPORTED"
    assert plan["sources"][0]["sidecar"]["sha256"] == sha256_file(path)


def test_plan_blocks_unsupported_wto_and_partial_qnl(tmp_path):
    _sidecar_for(tmp_path, "data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip",
                 "wto_public_corpus_en_fr", "https://www.wto.org/english/res_e/corpus_e/corpus_e.htm",
                 "WTO_PUBLIC_CORPUS_TERMS")
    _sidecar_for(tmp_path, "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip", "qnl_arabic_ocr_v2",
                 "https://manara.qnl.qa/articles/dataset/Arabic_OCR_Corpus_2_894_items_from_QNL_Collection_/26984785",
                 "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA")

    wto = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3", source_ids=["wto_public_corpus_en_fr"])
    assert wto["ready"] is False
    assert any("Generic TXT import" in blocker for blocker in wto["blockers"])

    qnl = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3", source_ids=["qnl_arabic_ocr_v2"])
    assert qnl["ready"] is False
    assert any("Generic ZIP/TXT import" in blocker for blocker in qnl["blockers"])

    qnl_allowed = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3",
                                   source_ids=["qnl_arabic_ocr_v2"], allow_partial=True)
    assert qnl_allowed["ready"] is True
    assert "limitation" in qnl_allowed["sources"][0]


def test_plan_refuses_frozen_tree_and_existing_output(tmp_path):
    _sidecar_for(tmp_path, "data/raw/external/tatoeba_cc0.tar.bz2", "tatoeba_cc0",
                 "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2", "CC0-1.0")
    frozen = repro.build_plan(tmp_path, output_dir=str(repro.PHASE10_2_PROCESSED), source_ids=["tatoeba_cc0"])
    assert frozen["ready"] is False
    assert any("frozen Phase 10.2" in blocker for blocker in frozen["blockers"])

    (tmp_path / "data/processed/phase10_3").mkdir(parents=True)
    existing = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3", source_ids=["tatoeba_cc0"])
    assert existing["ready"] is False
    assert any("already exists" in blocker for blocker in existing["blockers"])


def test_plan_reports_missing_sidecar(tmp_path):
    path = tmp_path / "data/raw/external/tatoeba_cc0.tar.bz2"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"no sidecar")
    plan = repro.build_plan(tmp_path, output_dir="data/processed/phase10_3", source_ids=["tatoeba_cc0"])
    assert plan["ready"] is False
    assert any("missing provenance sidecar" in blocker for blocker in plan["blockers"])


def test_split_stage_is_deterministic_for_fixed_records_and_seed(tmp_path):
    records = [
        {"text": f"Document {document} segment {segment} has distinct reviewable text.",
         "source_id": "fixture", "document_id": f"doc-{document}", "record_id": f"{document}-{segment}"}
        for document in range(6)
        for segment in range(4)
    ]
    first = split_records(records, tmp_path / "a", seed=10202, ratios={"train": 0.7, "validation": 0.15, "test": 0.15})
    second = split_records(records, tmp_path / "b", seed=10202, ratios={"train": 0.7, "validation": 0.15, "test": 0.15})
    assert first.split_counts == second.split_counts
    assert first.group_count == second.group_count
    for name in ("train", "validation", "test"):
        assert sha256_file(tmp_path / "a" / f"{name}.jsonl") == sha256_file(tmp_path / "b" / f"{name}.jsonl")


def test_cli_audit_writes_json_and_markdown(tmp_path):
    _write_jsonl(tmp_path / "data/processed/phase10_2/s1/train.jsonl",
                 [{"text": "x", "split_group_id": "g", "record_id": "r"}])
    exit_code = repro.main(["audit", "--root", str(tmp_path), "--no-hash",
                            "--json", "reports/audit.json", "--markdown", "reports/audit.md"])
    assert exit_code == 0
    payload = json.loads((tmp_path / "reports/audit.json").read_text(encoding="utf-8"))
    assert payload["summary"]["artifacts"] == 1
    assert "Phase 10.2 artifact audit" in (tmp_path / "reports/audit.md").read_text(encoding="utf-8")


def test_supported_pipeline_path_writes_sidecar_and_reproducible_splits(tmp_path):
    """End-to-end proof on a fixture: import -> normalize -> split with sidecars."""
    import io
    import shutil
    import tarfile

    from configs.configuration import load_config
    from corpus.pipeline import import_source_file, normalize_imported_file
    from data.splitting import split_sources
    from tokenizer.byte_tokenizer import ByteTokenizer

    rows = "".join(
        f"{index}\t{language}\t{text}\n"
        for index, (language, text) in enumerate(
            [("ara", "هذا نص عربي واضح للاختبار"), ("eng", "This is a clear English test sentence"),
             ("fra", "Ceci est une phrase française de test"), ("ara", "نص عربي ثان للاختبار"),
             ("eng", "Another English sentence for the fixture"), ("fra", "Une autre phrase française")],
            1,
        )
    ).encode("utf-8")
    archive = tmp_path / "data/raw/external/tatoeba_cc0.tar.bz2"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:bz2") as tar:
        info = tarfile.TarInfo("sentences_CC0.csv")
        info.size = len(rows)
        tar.addfile(info, io.BytesIO(rows))

    config = load_phase8_config()
    (tmp_path / "configs").mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "configs/default.json", tmp_path / "configs/default.json")
    meta = {"source_id": "tatoeba_cc0",
            "source_url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
            "license_id": "CC0-1.0", "bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
    archive.with_name(archive.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")

    imported = tmp_path / "data/processed/phase8/imported/imported.jsonl"
    stats = import_source_file("tatoeba_cc0", archive, imported, project_root=tmp_path, config=config)
    sidecar = imported.with_name(imported.name + ".import.json")
    assert sidecar.is_file()
    assert stats["accepted_records"] == 6
    assert json.loads(sidecar.read_text(encoding="utf-8"))["source_sha256"] == meta["sha256"]

    normalized = tmp_path / "data/processed/phase8/normalized/normalized.jsonl"
    normalize_imported_file(imported, normalized, project_root=tmp_path, config=config)
    project_config = load_config(tmp_path / "configs/default.json", ByteTokenizer())
    splits = tmp_path / "data/processed/phase8/splits"
    result = split_sources([normalized], splits, config=project_config,
                           ratios=config["split_ratios"], seed=config["random_seed"])
    assert sum(result.split_counts.values()) == 6
    for name in ("train", "validation", "test"):
        assert (splits / f"{name}.jsonl").is_file()

