from pathlib import Path

from corpus.integrity import (
    audit_artifact_integrity,
    inventory_orphan_and_frozen_artifacts,
    is_quarantined_or_frozen_path,
    verify_sqlite_files_table,
)


def test_quarantine_filter_identifies_orphans_and_frozen_phase10_2():
    root = Path(__file__).resolve().parents[1]
    assert is_quarantined_or_frozen_path(root, "data/processed/phase10_2/splits_v1/train.jsonl") is True
    assert is_quarantined_or_frozen_path(root, "data/processed/phase10_2/wto_en_fr/imported.jsonl.partial") is True
    assert is_quarantined_or_frozen_path(root, "data/raw/external/phase10_2/.parts/part01.part") is True
    assert is_quarantined_or_frozen_path(root, "reports/phase10/phase10_2/leakage-temp/seen.sqlite3") is True

    # Active Phase 10.3 and raw archives must NOT be quarantined
    assert is_quarantined_or_frozen_path(root, "data/processed/phase10_3/splits/train.jsonl") is False
    assert is_quarantined_or_frozen_path(root, "data/raw/external/phase10_2/qnl_arabic_ocr_v2.zip") is False
    assert is_quarantined_or_frozen_path(root, "data/raw/external/tatoeba_cc0.tar.bz2") is False


def test_sqlite_files_table_and_full_phase10_8_integrity_audit():
    root = Path(__file__).resolve().parents[1]
    sqlite_res = verify_sqlite_files_table(root, "data/metadata/phase10_3_corpus.sqlite3")
    assert sqlite_res["valid"] is True, sqlite_res["mismatches"]
    assert sqlite_res["files_checked"] == 12

    inv = inventory_orphan_and_frozen_artifacts(root)
    assert inv["files_deleted"] == 0
    assert inv["orphan_count"] >= 28  # 11 .partial/.err + 17 .parts + any temp scratch files
    assert inv["frozen_payload_count"] == 7  # 2 imported + 2 normalized + 3 splits_v1

    report = audit_artifact_integrity(root)
    assert report["status"] == "PASS", report["errors"]
    assert all(report["acceptance_criteria"].values())
