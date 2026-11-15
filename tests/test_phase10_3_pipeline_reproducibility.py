import bz2
import json
import shutil
from pathlib import Path

import pytest

from corpus import reproducibility as repro
from corpus.acquisition import sha256_file
from corpus.pipeline import import_source_file, load_phase8_config, normalize_imported_file, prepare_splits


def _setup_workspace(tmp_path):
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo_root / "configs/default.json", tmp_path / "configs/default.json")
    shutil.copyfile(repo_root / "configs/phase10_3_corpus.json", tmp_path / "configs/phase10_3_corpus.json")
    return load_phase8_config(tmp_path / "configs/phase10_3_corpus.json")


def _create_tatoeba_fixture(root, relative_path, rows):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{idx}\t{lang}\t{text}\t2025-01-01 00:00:00\n" for idx, lang, text in rows)
    path.write_bytes(bz2.compress(payload.encode("utf-8")))
    meta = {
        "source_id": "tatoeba_cc0",
        "source_url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
        "license_id": "CC0-1.0",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_phase10_3_config_loads_and_approves_all_three_sources():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_phase8_config(repo_root / "configs/phase10_3_corpus.json")
    assert cfg["approved_source_ids"] == ["tatoeba_cc0", "qnl_arabic_ocr_v2", "wto_public_corpus_en_fr"]
    assert cfg["split_ratios"] == {"train": 0.9, "validation": 0.05, "test": 0.05}
    assert cfg["random_seed"] == 1337
    assert cfg["splits_directory"] == "data/processed/phase10_3/splits"
    assert cfg["metadata_database"] == "data/metadata/phase10_3_corpus.sqlite3"


def test_pipeline_blocks_any_write_to_frozen_phase10_2(tmp_path):
    cfg = _setup_workspace(tmp_path)
    raw = _create_tatoeba_fixture(
        tmp_path,
        "data/raw/external/sentences_CC0.tsv.bz2",
        [(1, "eng", "Frozen protection test sentence.")],
    )
    with pytest.raises(PermissionError, match="frozen Phase 10.2"):
        import_source_file("tatoeba_cc0", raw, "data/processed/phase10_2/imported.jsonl", project_root=tmp_path, config=cfg)

    valid_import = import_source_file(
        "tatoeba_cc0", raw, "data/processed/phase10_3/imported/tatoeba.jsonl", project_root=tmp_path, config=cfg
    )
    with pytest.raises(PermissionError, match="frozen Phase 10.2"):
        normalize_imported_file(
            valid_import["imported_file"],
            "data/processed/phase10_2/normalized.jsonl",
            project_root=tmp_path,
            config=cfg,
        )

    bad_cfg = dict(cfg, splits_directory="data/processed/phase10_2/splits_v2")
    with pytest.raises(PermissionError, match="frozen Phase 10.2"):
        prepare_splits([valid_import["imported_file"]], project_root=tmp_path, config=bad_cfg)


def test_compact_traceable_normalization_and_audit_verdict(tmp_path):
    cfg = _setup_workspace(tmp_path)
    raw = _create_tatoeba_fixture(
        tmp_path,
        "data/raw/external/sentences_CC0.tsv.bz2",
        [
            (1, "ara", "نص عربي للاختبار الأول"),
            (2, "eng", "English sentence for compact normalization test"),
            (3, "fra", "Phrase française pour le test de normalisation"),
        ],
    )
    imp = import_source_file(
        "tatoeba_cc0", raw, "data/processed/phase10_3/imported/tatoeba.jsonl", project_root=tmp_path, config=cfg
    )
    norm_stats = normalize_imported_file(
        imp["imported_file"],
        "data/processed/phase10_3/normalized/tatoeba.normalized.jsonl",
        project_root=tmp_path,
        config=cfg,
        retain_stage_texts=False,
    )
    norm_path = Path(norm_stats["output_file"])
    sidecar_path = Path(norm_stats["sidecar_file"])
    assert norm_path.is_file() and sidecar_path.is_file()
    first_row = json.loads(norm_path.read_text(encoding="utf-8").splitlines()[0])
    assert "raw_text" not in first_row
    assert "normalized_text" not in first_row
    assert "cleaned_text" not in first_row
    assert {"raw_text_sha256", "normalized_text_sha256", "cleaned_text_sha256"} <= set(first_row.keys())

    audit = repro.audit_artifacts(tmp_path, targets=["data/processed/phase10_3"])
    by_path = {item["path"]: item for item in audit["artifacts"]}
    assert by_path["data/processed/phase10_3/imported/tatoeba.jsonl"]["reproducible"] == "YES"
    assert by_path["data/processed/phase10_3/normalized/tatoeba.normalized.jsonl"]["reproducible"] == "YES"


def test_reproducible_two_runs_and_run_manifest_verification(tmp_path):
    rows = []
    for i in range(1, 21):
        rows.append((i, "ara", f"هذه جملة عربية فريدة رقم {i} لاختبار إعادة الإنتاج الحتمي."))
        rows.append((100 + i, "eng", f"Unique English test sentence number {i} for deterministic pipeline reproduction."))
        rows.append((200 + i, "fra", f"Phrase française unique numéro {i} pour vérifier la reproductibilité."))

    cfg = _setup_workspace(tmp_path)
    raw = _create_tatoeba_fixture(tmp_path, "data/raw/external/sentences_CC0.tsv.bz2", rows)
    imp = import_source_file(
        "tatoeba_cc0", raw, "data/processed/phase10_3/imported/tatoeba.jsonl", project_root=tmp_path, config=cfg
    )
    run_hashes = []
    for sub in ("run_a", "run_b"):
        run_cfg = dict(
            cfg,
            normalized_directory=f"data/processed/phase10_3/{sub}/normalized",
            splits_directory=f"data/processed/phase10_3/{sub}/splits",
            metadata_database=f"data/metadata/phase10_3_{sub}.sqlite3",
        )
        res = prepare_splits(
            [imp["imported_file"]],
            project_root=tmp_path,
            config=run_cfg,
            retain_stage_texts=False,
            run_id=f"phase10_3_{sub}",
        )
        check = repro.verify_run_manifest(tmp_path, res.run_manifest_path, config=run_cfg, verify_hashes=True)
        assert check["valid"] is True, check["errors"]

        audit = repro.audit_artifacts(
            tmp_path,
            targets=[
                "data/processed/phase10_3/imported",
                f"data/processed/phase10_3/{sub}",
            ],
        )
        payload_verdicts = {
            item["path"]: item["reproducible"]
            for item in audit["artifacts"]
            if item["kind"] in {"imported", "normalized", "split"}
        }
        assert all(v == "YES" for v in payload_verdicts.values()), payload_verdicts

        split_rows = {}
        for name in ("train", "validation", "test"):
            spath = tmp_path / run_cfg["splits_directory"] / f"{name}.jsonl"
            parsed = [json.loads(line) for line in spath.read_text(encoding="utf-8").splitlines() if line.strip()]
            for item in parsed:
                item.pop("source", None)
            split_rows[name] = parsed
        run_hashes.append(split_rows)

    assert run_hashes[0] == run_hashes[1]


def test_verify_run_manifest_detects_tampering_and_bad_split_decisions(tmp_path):
    cfg = _setup_workspace(tmp_path)
    rows = [(i, "eng", f"Tamper check sentence {i} in English.") for i in range(1, 25)]
    raw = _create_tatoeba_fixture(tmp_path, "data/raw/external/sentences_CC0.tsv.bz2", rows)
    imp = import_source_file(
        "tatoeba_cc0", raw, "data/processed/phase10_3/imported/tatoeba.jsonl", project_root=tmp_path, config=cfg
    )
    res = prepare_splits([imp["imported_file"]], project_root=tmp_path, config=cfg, run_id="tamper_test")
    manifest_path = Path(res.run_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Tamper with split decision seed in manifest
    bad_manifest = dict(manifest, split_decisions={"requested_ratios": cfg["split_ratios"], "seed": 9999, "grouping_field": "document_id"})
    verdict = repro.verify_run_manifest(tmp_path, bad_manifest, config=cfg)
    assert verdict["valid"] is False
    assert any("random_seed" in err for err in verdict["errors"])

    # Tamper with train.jsonl bytes
    train_path = tmp_path / "data/processed/phase10_3/splits/train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    verdict_hash = repro.verify_run_manifest(tmp_path, manifest_path, config=cfg, verify_hashes=True)
    assert verdict_hash["valid"] is False
    assert any("split sha256 mismatch for train" in err for err in verdict_hash["errors"])


def test_multi_source_prepare_splits_and_manifest(tmp_path):
    cfg = _setup_workspace(tmp_path)
    raw_tatoeba = _create_tatoeba_fixture(
        tmp_path,
        "data/raw/external/sentences_CC0.tsv.bz2",
        [(i, "eng", f"Tatoeba multi-source sentence {i}.") for i in range(1, 21)],
    )
    imp_tatoeba = import_source_file(
        "tatoeba_cc0", raw_tatoeba, "data/processed/phase10_3/imported/tatoeba.jsonl", project_root=tmp_path, config=cfg
    )

    raw_qnl = tmp_path / "data/raw/external/qnl_fixture.txt"
    raw_qnl.parent.mkdir(parents=True, exist_ok=True)
    qnl_lines = "".join(f"وثيقة عربية رقم {i} من مكتبة قطر الوطنية للاختبار.\n" for i in range(1, 21))
    raw_qnl.write_text(qnl_lines, encoding="utf-8")
    qnl_meta = {
        "source_id": "qnl_arabic_ocr_v2",
        "official_item_url": "https://manara.qnl.qa/articles/dataset/Arabic_OCR_Corpus_2_894_items_from_QNL_Collection_/26984785",
        "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA",
        "bytes": raw_qnl.stat().st_size,
        "sha256": sha256_file(raw_qnl),
    }
    raw_qnl.with_name(raw_qnl.name + ".provenance.json").write_text(json.dumps(qnl_meta), encoding="utf-8")
    imp_qnl = import_source_file(
        "qnl_arabic_ocr_v2", raw_qnl, "data/processed/phase10_3/imported/qnl.jsonl", project_root=tmp_path, config=cfg
    )

    res = prepare_splits(
        [imp_tatoeba["imported_file"], imp_qnl["imported_file"]],
        project_root=tmp_path,
        config=cfg,
        retain_stage_texts=False,
        run_id="multi_source_test_run",
    )
    assert res.unique_records == 40
    check = repro.verify_run_manifest(tmp_path, res.run_manifest_path, config=cfg, verify_hashes=True)
    assert check["valid"] is True, check["errors"]
    manifest_data = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
    assert manifest_data["sources"] == ["tatoeba_cc0", "qnl_arabic_ocr_v2"]

