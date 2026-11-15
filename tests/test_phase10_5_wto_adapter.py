import json
from pathlib import Path
import shutil
import zipfile

import pytest

from configs.configuration import load_config
from corpus.acquisition import sha256_file
from corpus.adapters.wto_adapter import (
    DEFAULT_WTO_MEMBER,
    EXPECTED_WTO_HEADER,
    WTO_ADAPTER_VERSION,
    WTOAdapterError,
    iter_wto_archive_records,
    scan_wto_member_domains,
    select_domain_stratified_documents,
)
from corpus.pipeline import import_source_file, load_phase8_config, normalize_imported_file
from corpus.reproducibility import audit_artifacts, build_plan
from data.splitting import split_sources
from data.validation import validate_splits
from tokenizer.byte_tokenizer import ByteTokenizer


def _build_wto_fixture_zip(zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(EXPECTED_WTO_HEADER) + "\r\n"
    # Row 1: doc 1001, seg 1, domain AG, contains embedded continuation newline in SegSrc
    row1 = (
        "600001\t1001\t1\t2018\t\\G\\AG\\\tW1.doc\t"
        "First line of English segment\ncontinued on second line.\t"
        "Première ligne du segment français continuée.\t1-1\tAG\r\n"
    )
    # Row 2: doc 1001, seg 2, domain AG, normal single-line
    row2 = (
        "600002\t1001\t2\t2018\t\\G\\AG\\\tW1.doc\t"
        "Second English segment in document 1001.\t"
        "Deuxième segment français dans le document 1001.\t1-1\tAG\r\n"
    )
    # Row 3: doc 1002, seg 1, domain AG, unpaired (empty French target) -> must be skipped
    row3 = (
        "600003\t1002\t1\t2018\t\\G\\AG\\\tW2.doc\t"
        "Unpaired English segment only.\t   \t1-0\tAG\r\n"
    )
    # Row 4: doc 2001, seg 1, domain SPS
    row4 = (
        "600004\t2001\t1\t2019\t\\G\\SPS\\\tN10.doc\t"
        "Sanitary and phytosanitary notification text in English.\t"
        "Texte de notification sanitaire et phytosanitaire en français.\t1-1\tSPS\r\n"
    )
    # Row 5: doc 3001, seg 1, empty domain -> UNKNOWN
    row5 = (
        "600005\t3001\t1\t2017\t\\WT\\DS\\\tR5.doc\t"
        "Dispute settlement panel report paragraph one.\t"
        "Rapport du groupe spécial de règlement des différends paragraphe un.\t1-1\t\r\n"
    )
    payload = (header + row1 + row2 + row3 + row4 + row5).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DEFAULT_WTO_MEMBER, payload)

    sidecar = {
        "source_id": "wto_public_corpus_en_fr",
        "source_url": "https://www.wto.org/english/res_e/corpus_e/corpus_e.htm",
        "license_id": "WTO_PUBLIC_CORPUS_TERMS",
        "bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "retrieved_at": "2026-11-13T00:00:00+00:00",
    }
    zip_path.with_name(zip_path.name + ".provenance.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return zip_path


def test_wto_continuation_line_reconstruction_and_unpaired_filtering(tmp_path):
    zip_path = _build_wto_fixture_zip(tmp_path / "wto.zip")
    scan = scan_wto_member_domains(zip_path)
    assert scan["records_in_first_pass"] == 5
    assert scan["unpaired_rows_skipped"] == 1
    assert scan["paired_rows_available"] == 4
    assert scan["domain_to_docs"] == {
        "AG": ["1001", "1002"],
        "SPS": ["2001"],
        "UNKNOWN": ["3001"],
    }

    records = list(iter_wto_archive_records(zip_path, sampling_mode="stratified_chunk00", seed=1337))
    # 4 paired rows -> 8 records (4 eng + 4 fra)
    assert len(records) == 8
    first_eng, first_fra = records[0], records[1]
    assert first_eng["source_record_id"] == "600001:eng"
    assert first_fra["source_record_id"] == "600001:fra"
    assert first_eng["document_id"] == "1001"
    assert first_fra["document_id"] == "1001"
    assert first_eng["translation_group_id"] == "1001:1"
    assert first_fra["translation_group_id"] == "1001:1"
    assert "\ncontinued on second line." in first_eng["text"]
    assert first_eng["adapter_version"] == WTO_ADAPTER_VERSION


def test_wto_deterministic_domain_stratified_sampling():
    chunk00_counts = {
        "ACC": 219,
        "ADP": 192,
        "AFT": 2,
        "AG": 247,
        "BFA": 62,
        "DS": 308,
        "MIN": 156,
        "OFFICE": 39,
        "RL": 18,
        "SCM": 223,
        "SG": 122,
        "SPS": 900,
        "TBT": 1345,
        "TF": 5,
        "TPR": 138,
        "UNKNOWN": 1447,
    }
    synthetic_domain_docs = {
        dom: [f"{dom}_{i}" for i in range(count)]
        for dom, count in chunk00_counts.items()
    }
    selected_a, summary_a = select_domain_stratified_documents(synthetic_domain_docs, seed=1337)
    selected_b, summary_b = select_domain_stratified_documents(synthetic_domain_docs, seed=1337)
    assert selected_a == selected_b
    assert summary_a["selected_documents"] == 124
    assert summary_a["domain_document_sampling"]["TBT"]["selected_documents"] == 27
    assert summary_a["domain_document_sampling"]["UNKNOWN"]["selected_documents"] == 29
    assert summary_a["domain_document_sampling"]["AFT"]["selected_documents"] == 2


def test_wto_pipeline_import_normalize_and_split_group_integrity(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_root / "configs/default.json", tmp_path / "configs/default.json")
    shutil.copyfile(repo_root / "configs/phase10_3_corpus.json", tmp_path / "configs/phase10_3_corpus.json")

    zip_path = _build_wto_fixture_zip(tmp_path / "data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip")
    config = load_phase8_config(tmp_path / "configs/phase10_3_corpus.json")

    imported_path = tmp_path / "data/processed/phase10_3/imported/wto_public_corpus_en_fr.jsonl"
    stats = import_source_file(
        "wto_public_corpus_en_fr",
        zip_path,
        imported_path,
        project_root=tmp_path,
        config=config,
    )
    assert stats["accepted_records"] == 8
    assert stats["adapter_version"] == WTO_ADAPTER_VERSION
    assert stats["languages"] == {"eng": 4, "fra": 4}

    normalized_path = tmp_path / "data/processed/phase10_3/normalized/wto_public_corpus_en_fr.normalized.jsonl"
    normalize_imported_file(
        imported_path,
        normalized_path,
        project_root=tmp_path,
        config=config,
        retain_stage_texts=False,
    )

    splits_dir = tmp_path / "data/processed/phase10_3/splits"
    project_config = load_config(tmp_path / "configs/default.json", ByteTokenizer())
    split_sources(
        [normalized_path],
        splits_dir,
        config=project_config,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=1337,
        group_field="document_id",
    )

    split_paths = {name: splits_dir / f"{name}.jsonl" for name in ("train", "validation", "test")}
    validation = validate_splits(split_paths, temp_directory=tmp_path / "tmp_val")
    assert validation["valid"] is True, validation["errors"]
    assert validation["duplicate_leakage"] == 0
    assert validation["document_group_leakage"] == 0

    # Verify every translation_group_id stays within a single split
    group_to_splits = {}
    for split_name in ("train", "validation", "test"):
        split_file = splits_dir / f"{split_name}.jsonl"
        for line in split_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            tg_id = row["translation_group_id"]
            group_to_splits.setdefault(tg_id, set()).add(split_name)
    assert all(len(splits) == 1 for splits in group_to_splits.values())

    audit = audit_artifacts(tmp_path, targets=["data/processed/phase10_3"])
    by_path = {item["path"]: item for item in audit["artifacts"]}
    assert by_path["data/processed/phase10_3/imported/wto_public_corpus_en_fr.jsonl"]["reproducible"] == "YES"
    assert by_path["data/processed/phase10_3/normalized/wto_public_corpus_en_fr.normalized.jsonl"]["reproducible"] == "YES"


def test_real_wto_archive_plan_and_streaming_verification():
    repo_root = Path(__file__).resolve().parents[1]
    plan = build_plan(
        repo_root,
        output_dir="data/processed/phase10_3_dryrun_check",
        source_ids=["tatoeba_cc0", "qnl_arabic_ocr_v2", "wto_public_corpus_en_fr"],
        verify_hashes=False,
    )
    assert plan["ready"] is True, plan["blockers"]
    by_source = {s["source_id"]: s for s in plan["sources"]}
    assert by_source["wto_public_corpus_en_fr"]["support"]["support"] == "SUPPORTED"

    wto_zip = repo_root / "data/raw/external/phase10_2/wto_en_fr_corpus_2019_EN-FR.zip"
    if not wto_zip.is_file():
        pytest.skip("Real WTO archive not present")
    stream = iter_wto_archive_records(wto_zip, sampling_mode="full")
    sample = [next(stream) for _ in range(6)]
    assert len(sample) == 6
    assert sample[0]["source_language_code"] == "eng"
    assert sample[1]["source_language_code"] == "fra"
    assert sample[0]["translation_group_id"] == sample[1]["translation_group_id"]
    assert sample[0]["document_id"] == sample[1]["document_id"]
    assert sample[0]["adapter_version"] == WTO_ADAPTER_VERSION
