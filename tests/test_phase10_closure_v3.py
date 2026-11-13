from __future__ import annotations
import csv
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/phase10/final_closure_v3'


def test_phase10_v3_sqlite_migration_integrity_and_provenance():
    report = json.loads((OUT / 'migration_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'MIGRATED_AND_VERIFIED'
    assert report['mapped_rows'] == 55716
    assert report['rollback_test'] == 'PASS'
    db = ROOT / 'data/metadata/corpus.sqlite3'
    con = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        assert con.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
        assert con.execute('PRAGMA foreign_key_check').fetchall() == []
        assert con.execute("SELECT COUNT(*) FROM records WHERE split IS NOT NULL AND json_extract(metadata_json,'$.source_record_id_migration.status')='VERIFIED'").fetchone()[0] == 55716
        assert con.execute("SELECT COUNT(*) FROM records WHERE split IS NOT NULL AND (source_record_id LIKE 'train:%' OR source_record_id LIKE 'validation:%' OR source_record_id LIKE 'test:%')").fetchone()[0] == 0
    finally:
        con.close()


def test_phase10_v3_review_decisions_are_resolved_and_dry_run_only():
    with (OUT / 'review_pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    assert sum(r['pair_type'] == 'near_duplicate_cross_split' for r in rows) == 12
    assert sum(r['pair_type'] == 'normalized_duplicate' for r in rows) == 3
    assert all(r['decision'].strip() for r in rows)
    result = subprocess.run([sys.executable, str(OUT / 'apply_review_decisions.py')], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload['mode'] == 'DRY_RUN'
    assert payload['pending'] == 0
    assert payload['validated_decisions'] == 15
    assert payload['writes_performed'] is False
    assert (OUT / 'reviewed_splits').is_dir()


def test_phase10_v3_document_group_gate_fails_closed():
    policy = json.loads((OUT / 'document_leakage_policy.json').read_text(encoding='utf-8'))
    assert policy['status'] == 'NOT_CHECKABLE'
    assert policy['translation_links_export_present'] is False
    assert policy['future_acceptance_gate']['document_id_or_group_id_required_when_source_has_document_or_translation_group_structure'] is True
    assert policy['future_acceptance_gate']['missing_value_action'].startswith('BLOCKED or UNKNOWN')
