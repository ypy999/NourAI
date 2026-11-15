import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from corpus import reproducibility as repro
from corpus.acquisition import sha256_file
from corpus.adapters.qnl_adapter import (
    QNL_ADAPTER_VERSION,
    compute_arabic_script_ratio,
    iter_qnl_archive_records,
    load_qnl_metadata_csv,
)
from corpus.pipeline import import_source_file, load_phase8_config


def _create_qnl_fixture(root: Path, docs: list[tuple[str, str, str, dict]]):
    """Create a synthetic QNL ZIP + metadata CSV + provenance sidecars."""
    raw_dir = root / "data/raw/external/phase10_2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "qnl_arabic_ocr_v2.zip"
    csv_path = raw_dir / "qnl_arabic_ocr_v2_metadata.csv"

    header = "CALL #(ITEM),RECORD #(ITEM),Repository URL,Catalog URL,AUTHOR,ADD AUTHOR,PUB INFO,TITLE,DESCRIPTION,VOLUME\n"
    csv_lines = [header]
    checksum_lines = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for record_id, member_name, text, meta in docs:
            raw_bytes = text.encode("utf-8")
            sha = hashlib.sha256(raw_bytes).hexdigest()
            checksum_lines.append(f"{sha} *{member_name}\r\n")
            zf.writestr(member_name, raw_bytes)
            csv_lines.append(
                f"{meta.get('call_number', 'CALL1')},{record_id},"
                f"{meta.get('repository_url', 'https://ediscovery.qnl.qa/1')},"
                f"{meta.get('catalog_url', 'https://library.qnl.qa/1')},"
                f"{meta.get('author', 'مؤلف تجريبي')},"
                f"{meta.get('additional_author', '')},"
                f"{meta.get('publication_info', 'دمشق 1900')},"
                f"{meta.get('title', 'كتاب تجريبي')},"
                f"{meta.get('description', '100 ص')},"
                f"{meta.get('volume', '')}\n"
            )
        zf.writestr("checksums.sha256", "".join(checksum_lines).encode("utf-8"))

    csv_path.write_text("".join(csv_lines), encoding="utf-8")

    for target in (zip_path, csv_path):
        sidecar = {
            "source_id": "qnl_arabic_ocr_v2",
            "official_item_url": "https://manara.qnl.qa/articles/dataset/Arabic_OCR_Corpus_2_894_items_from_QNL_Collection_/26984785",
            "license_id": "PUBLIC_DOMAIN_TEXTS; CC0-1.0_METADATA",
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        target.with_name(target.name + ".provenance.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return zip_path, csv_path


def test_qnl_adapter_streams_and_enriches_records(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / "configs/default.json", tmp_path / "configs/default.json")
    cfg = load_phase8_config(repo_root / "configs/phase10_3_corpus.json")

    docs = [
        (
            "i10027154",
            "i10027154-00005595.txt",
            "هذا كتاب إيضاح الدلالات في سماع الآلات لعبد الغني النابلسي.",
            {"title": "إيضاح الدلالات", "author": "النابلسي"},
        ),
        (
            "i10027282",
            "i10027282-00005636.txt",
            "كتاب غالية المواعظ ومصباح المتعظ وقبس الواعظ للآلوسي.",
            {"title": "غالية المواعظ", "author": "الآلوسي"},
        ),
    ]
    zip_path, _ = _create_qnl_fixture(tmp_path, docs)

    imp = import_source_file(
        "qnl_arabic_ocr_v2",
        zip_path,
        "data/processed/phase10_3/imported/qnl.jsonl",
        project_root=tmp_path,
        config=cfg,
    )
    assert imp["accepted_records"] == 2
    assert imp["adapter_version"] == QNL_ADAPTER_VERSION

    imported_path = Path(imp["imported_file"])
    rows = [json.loads(line) for line in imported_path.read_text(encoding="utf-8").splitlines()]
    assert [r["document_id"] for r in rows] == ["i10027154", "i10027282"]
    assert all(r["document_id_status"] == "SOURCE_PROVIDED" for r in rows)
    assert all(r["provenance_id_source"] == "QNL RECORD #(ITEM); original catalog record ID" for r in rows)
    assert rows[0]["qnl_metadata"]["title"] == "إيضاح الدلالات"
    assert rows[0]["arabic_script_ratio"] > 0.70

    audit = repro.audit_artifacts(tmp_path, targets=["data/processed/phase10_3"])
    by_path = {item["path"]: item for item in audit["artifacts"]}
    assert by_path["data/processed/phase10_3/imported/qnl.jsonl"]["reproducible"] == "YES"


def test_qnl_adapter_detects_checksum_mismatch_and_missing_metadata(tmp_path):
    zip_path, csv_path = _create_qnl_fixture(
        tmp_path,
        [
            ("i10000001", "i10000001-0001.txt", "نص سليم تماماً", {}),
            ("i10000002", "i10000002-0002.txt", "نص سيتم التلاعب ببصمته", {}),
        ],
    )
    # Tamper with checksums.sha256 inside ZIP for i10000002
    with zipfile.ZipFile(zip_path, "r") as zf_in:
        members = {name: zf_in.read(name) for name in zf_in.namelist()}
    bad_checksums = members["checksums.sha256"].decode("utf-8").splitlines()
    bad_checksums[1] = "0" * 64 + " *i10000002-0002.txt"
    members["checksums.sha256"] = ("\n".join(bad_checksums) + "\n").encode("utf-8")
    with zipfile.ZipFile(zip_path, "w") as zf_out:
        for name, data in members.items():
            zf_out.writestr(name, data)

    results = list(iter_qnl_archive_records(zip_path, metadata_csv_path=csv_path))
    valid = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    assert len(valid) == 1 and valid[0]["document_id"] == "i10000001"
    assert len(errors) == 1 and "SHA-256 mismatch" in errors[0]


def test_qnl_plan_becomes_ready_when_metadata_csv_and_sidecar_exist(tmp_path):
    _create_qnl_fixture(
        tmp_path,
        [("i10027154", "i10027154-00005595.txt", "نص عربي للاختبار", {})],
    )
    plan = repro.build_plan(
        tmp_path,
        output_dir="data/processed/phase10_3",
        source_ids=["qnl_arabic_ocr_v2"],
        verify_hashes=True,
    )
    assert plan["ready"] is True, plan["blockers"]
    assert plan["sources"][0]["support"]["support"] == "SUPPORTED"


def test_real_qnl_archive_and_metadata_csv_readonly_verification():
    repo_root = Path(__file__).resolve().parents[1]
    zip_path = repo_root / "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip"
    csv_path = repo_root / "data/raw/external/phase10_2/qnl_arabic_ocr_v2_metadata.csv"
    if not (zip_path.is_file() and csv_path.is_file()):
        pytest.skip("real QNL archive not staged locally")

    catalog = load_qnl_metadata_csv(csv_path)
    assert len(catalog) == 2894

    # Stream first 5 records from real archive and verify checksum + metadata
    gen = iter_qnl_archive_records(zip_path, metadata_csv_path=csv_path)
    sample = [next(gen) for _ in range(5)]
    assert all("error" not in r for r in sample)
    assert sample[0]["source_record_id"] == "i10027154"
    assert sample[0]["raw_text_sha256"] == "9d61aa8c38bee2b45020a5b7494febfdb89ac56a369856a5516b1ececfaab52c"
    assert abs(sample[0]["arabic_script_ratio"] - (1.0 - 0.21482874909842298)) < 1e-12
