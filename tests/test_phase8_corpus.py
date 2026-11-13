"""Phase 8 source, license, streaming import, safety and readiness checks."""
import bz2
import io
import json
import tarfile
import zipfile

import pytest

from corpus.acquisition import AcquisitionError, download_resumable, sha256_file
from corpus.benchmark import benchmark_jsonl, benchmark_source_records
from corpus.importers import iter_source_records
from corpus.licensing import require_approved_source, source_use_status
from corpus.pipeline import corpus_statistics, import_source_file, load_phase8_config
from corpus.pipeline import prepare_splits
from corpus.readiness import assess_readiness
from corpus.reporting import build_report, write_dashboard, write_report
from corpus.storage import StorageManager
from corpus.validation import validate_jsonl, validate_splits
from tokenizer.byte_tokenizer import ByteTokenizer


def test_license_catalog_allows_only_explicit_cc0():
    assert source_use_status("tatoeba_cc0")["approved_for_automatic_ingestion"]
    for source in ("tatoeba_general", "wikimedia", "oscar"):
        assert not source_use_status(source)["approved_for_automatic_ingestion"]
        with pytest.raises(PermissionError):
            require_approved_source(source)


def test_streaming_imports_txt_jsonl_and_csv(tmp_path):
    txt = tmp_path / "source.txt"
    txt.write_text("مرحبا\nbonjour\nhello\n", encoding="utf-8")
    assert [r["text"] for r in iter_source_records(txt)] == ["مرحبا", "bonjour", "hello"]
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"text":"مرحبا"}\n{"content":"bonjour"}\ninvalid\n', encoding="utf-8")
    rows = list(iter_source_records(jsonl))
    assert [r["text"] for r in rows if "text" in r] == ["مرحبا", "bonjour"]
    assert any("error" in r for r in rows)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("text,lang\nhello,en\n", encoding="utf-8")
    assert next(iter_source_records(csv_path))["text"] == "hello"


def test_tatoeba_cc0_tsv_and_bz2(tmp_path):
    raw = "1\tara\tمرحبا\n2\teng\thello\n"
    compressed = tmp_path / "sentences_CC0.tsv.bz2"
    compressed.write_bytes(bz2.compress(raw.encode("utf-8")))
    rows = list(iter_source_records(compressed))
    assert rows[0]["text"] == "مرحبا" and rows[0]["license_id"] == "CC0-1.0"
    assert rows[1]["language_code"] == "eng"


def test_streaming_json_array_and_optional_parquet(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"text": "مرحبا"}, {"content": "bonjour"}], ensure_ascii=False), encoding="utf-8")
    assert [r["text"] for r in iter_source_records(path)] == ["مرحبا", "bonjour"]
    parquet = tmp_path / "data.parquet"
    parquet.write_bytes(b"not parquet")
    parsed = list(iter_source_records(parquet))
    if "requires optional pyarrow" in parsed[0].get("error", ""):
        assert "optional pyarrow" in parsed[0]["error"]
    else:
        assert any("Parquet read failed" in row.get("error", "") for row in parsed)


def test_tar_bz2_archive_streams_without_extracting(tmp_path):
    archive_path = tmp_path / "sentences_CC0.tar.bz2"
    source = io.BytesIO(b"1\tfra\tbonjour\t2025-01-02 00:00:00\n")
    with tarfile.open(archive_path, "w:bz2") as archive:
        info = tarfile.TarInfo("sentences_CC0.csv")
        info.size = len(source.getvalue())
        archive.addfile(info, source)
    rows = list(iter_source_records(archive_path))
    assert rows[0]["text"] == "bonjour"
    assert rows[0]["contributor_timestamp"] == "2025-01-02 00:00:00"


def test_zip_path_safety_and_streaming(tmp_path):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("records.txt", "hello\n")
        archive.writestr("../escape.txt", "no\n")
    rows = list(iter_source_records(path))
    assert any(row.get("text") == "hello" for row in rows)
    assert any("unsafe ZIP path" in row.get("error", "") for row in rows)


def test_zip_expansion_total_limit(tmp_path):
    path = tmp_path / "bounded.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("one.txt", "1234\n")
        archive.writestr("two.txt", "5678\n")
    rows = list(iter_source_records(path, max_archive_total_bytes=6))
    assert any("total size limit" in row.get("error", "") for row in rows)


def test_invalid_utf8_and_oversized_line_are_reported(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"valid\n\xffbad\n")
    assert any("invalid UTF-8" in row.get("error", "") for row in iter_source_records(bad))
    huge = tmp_path / "huge.txt"
    huge.write_bytes(b"x" * (8 * 1024 * 1024 + 1) + b"\n")
    assert any("safety limit" in row.get("error", "") for row in iter_source_records(huge))
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip archive")
    assert any("corrupt or unreadable" in row.get("error", "") for row in iter_source_records(corrupt))


def test_import_requires_known_source_and_preserves_raw(tmp_path):
    raw = tmp_path / "sentences_CC0.tsv.bz2"
    raw.write_bytes(bz2.compress("1\tara\tالسَّلَامُ 👋\n2\teng\tHello!\n3\tfra\tBonjour !\n4\tdeu\tHallo\n".encode()))
    sidecar = {"source_id": "tatoeba_cc0", "source_url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
               "license_id": "CC0-1.0", "bytes": raw.stat().st_size, "sha256": sha256_file(raw),
               "retrieved_at": "2026-01-01T00:00:00+00:00"}
    raw.with_name(raw.name + ".provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")
    output = tmp_path / "imported.jsonl"
    stats = import_source_file("tatoeba_cc0", raw, output, project_root=tmp_path)
    assert raw.exists()
    assert stats["accepted_records"] == 3
    assert stats["filtered_language_records"] == 1
    parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert parsed[0]["text"] == "السَّلَامُ 👋" and parsed[0]["license_id"] == "CC0-1.0"
    assert import_source_file("tatoeba_cc0", raw, output, project_root=tmp_path)["already_present"] is True
    with pytest.raises(PermissionError):
        import_source_file("oscar", raw, tmp_path / "bad.jsonl", project_root=tmp_path)


def test_import_rejects_missing_or_tampered_provenance(tmp_path):
    raw = tmp_path / "source.tsv.bz2"
    raw.write_bytes(bz2.compress(b"1\teng\tHello\n"))
    with pytest.raises(Exception, match="provenance"):
        import_source_file("tatoeba_cc0", raw, tmp_path / "first.jsonl", project_root=tmp_path)
    sidecar = {"source_id": "tatoeba_cc0", "source_url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
               "license_id": "CC0-1.0", "bytes": raw.stat().st_size, "sha256": "0" * 64}
    raw.with_name(raw.name + ".provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(Exception, match="SHA-256"):
        import_source_file("tatoeba_cc0", raw, tmp_path / "second.jsonl", project_root=tmp_path)


def test_import_to_phase7_split_integration(tmp_path):
    import shutil
    from pathlib import Path

    root = tmp_path
    (root / "configs").mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "configs" / "default.json", root / "configs" / "default.json")
    raw = root / "data" / "raw" / "sentences_CC0.tsv.bz2"
    raw.parent.mkdir(parents=True)
    lines = "".join(f"{i}\teng\tSentence {i} is unique for integration.\t2025-01-01 00:00:00\n" for i in range(1, 61))
    raw.write_bytes(bz2.compress(lines.encode("utf-8")))
    meta = {"source_id": "tatoeba_cc0", "source_url": "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
            "license_id": "CC0-1.0", "bytes": raw.stat().st_size, "sha256": sha256_file(raw)}
    raw.with_name(raw.name + ".provenance.json").write_text(json.dumps(meta), encoding="utf-8")
    imported = import_source_file("tatoeba_cc0", raw, "data/processed/import.jsonl", project_root=root)
    assert imported["accepted_records"] == 60
    cfg = dict(load_phase8_config(), splits_directory="data/processed/splits")
    result = prepare_splits([root / "data/processed/import.jsonl"], project_root=root, config=cfg)
    assert result.unique_records == 60
    split_root = root / "data/processed/splits"
    checked = validate_splits({name: split_root / f"{name}.jsonl" for name in ("train", "validation", "test")},
                              temp_directory=root / "data/processed")
    assert checked["valid"] and checked["duplicate_leakage"] == 0


def test_tokenizer_multilingual_round_trip():
    tokenizer = ByteTokenizer()
    examples = ["السلام عليكم", "English text with punctuation!", "Ça va très bien, merci.",
                "Mixed العربية and English", "12345 — € # @", "👨‍👩‍👧‍👦 🚀"]
    for text in examples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_import_checkpoint_resumes_without_duplicate_rows(tmp_path, monkeypatch):
    import corpus.pipeline as pipeline_module

    source = tmp_path / "sentences_CC0.tsv.bz2"
    source.write_bytes(bz2.compress(b"fixture source bytes"))
    sidecar = {"source_id":"tatoeba_cc0","source_url":"https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2",
               "license_id":"CC0-1.0","bytes":source.stat().st_size,"sha256":sha256_file(source)}
    source.with_name(source.name+".provenance.json").write_text(json.dumps(sidecar),encoding="utf-8")
    rows=[{"sentence_id":str(i),"language_code":"eng","text":f"Sentence {i} is valid.","record_number":i} for i in range(4)]
    original=pipeline_module.iter_source_records
    calls={"count":0}
    def interrupted(*args,**kwargs):
        calls["count"]+=1
        if calls["count"]==1:
            yield from rows[:3]
            raise RuntimeError("test interruption")
        yield from rows
    monkeypatch.setattr(pipeline_module,"iter_source_records",interrupted)
    config=dict(load_phase8_config(),batch_records=2)
    output=tmp_path/"imported.jsonl"
    with pytest.raises(RuntimeError,match="interruption"):
        import_source_file("tatoeba_cc0",source,output,project_root=tmp_path,config=config)
    assert output.with_name(output.name+".part").exists()
    monkeypatch.setattr(pipeline_module,"iter_source_records",lambda *args,**kwargs: iter(rows))
    result=import_source_file("tatoeba_cc0",source,output,project_root=tmp_path,config=config,resume=True)
    assert result["accepted_records"]==4
    parsed=[json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["source_record_id"] for row in parsed]==["0","1","2","3"]
    assert all("raw_text_sha256" in row and "language_estimate" in row for row in parsed)


def test_storage_blocks_path_escape_and_overwrite(tmp_path):
    manager = StorageManager(tmp_path)
    with pytest.raises(ValueError):
        manager.resolve("../outside")
    path = manager.atomic_jsonl("data/new.jsonl", [{"text": "x"}])
    assert path.exists()
    with pytest.raises(FileExistsError):
        manager.atomic_jsonl("data/new.jsonl", [])


def test_validate_jsonl_provenance_and_unicode(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(json.dumps({"text": "مرحبا 👋", "source_id": "tatoeba_cc0", "license_id": "CC0-1.0"}) + "\n", encoding="utf-8")
    result = validate_jsonl(path, require_provenance=True)
    assert result["valid"] and result["records"] == 1
    path.write_bytes(b'{"text":"x","source_id":"oscar","license_id":"CC0-1.0"}\n')
    assert not validate_jsonl(path, require_provenance=True)["valid"]


def test_validate_splits_detects_leakage_and_empty(tmp_path):
    paths = {}
    for split, text in (("train", "hello"), ("validation", " HELLO "), ("test", "bonjour")):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
        paths[split] = path
    result = validate_splits(paths, temp_directory=tmp_path)
    assert not result["valid"] and result["duplicate_leakage"] == 1


def test_validate_splits_detects_duplicates_inside_split(tmp_path):
    paths = {}
    for split, texts in (("train", ["same", "SAME"]), ("validation", ["different"]), ("test", ["third"])):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
        paths[split] = path
    result = validate_splits(paths, temp_directory=tmp_path)
    assert not result["valid"] and result["duplicates_within_splits"] == 1


def test_readiness_has_no_size_gate_and_blocks_unreviewed_license():
    result = assess_readiness(sources=[{"source_id": "oscar", "status": "UNKNOWN_LICENSE", "license_id": "unknown"}],
                              imports=[{"accepted_records": 1000000}],
                              splits={"train": 1, "validation": 1, "test": 1}, quality={"overall_dataset_score": 99},
                              validation={"valid": True})
    assert result["status"] == "NOT_READY" and result["size_threshold_used"] is False
    assert any("not approved" in blocker for blocker in result["blockers"])


def test_reports_json_markdown_and_dashboard_are_non_overwriting(tmp_path):
    report = build_report(environment={"python": "test"}, sources=[], downloads=[], imports=[], splits={},
                          validation={}, quality={}, performance={}, readiness={"status": "NOT_READY", "blockers": []})
    paths = write_report(report, tmp_path / "phase8.json")
    assert __import__("json").loads(open(paths["json"], encoding="utf-8").read())["claims"]["training_started"] is False
    assert "No model training" in open(paths["markdown"], encoding="utf-8").read()
    dashboard = write_dashboard(report, tmp_path / "index.html")
    assert "Corpus Readiness" in open(dashboard, encoding="utf-8").read()
    with pytest.raises(FileExistsError):
        write_report(report, tmp_path / "phase8.json")
    with pytest.raises(FileExistsError):
        write_dashboard(report, tmp_path / "index.html")
    assert write_dashboard(report, tmp_path / "index.html", overwrite=True)


def test_phase8_config_validates_known_sources():
    config = load_phase8_config()
    assert config["approved_source_ids"] == ["tatoeba_cc0"]


def test_phase8_config_rejects_invalid_split_ratios(tmp_path):
    config = load_phase8_config()
    config["split_ratios"] = {"train": 0.8, "validation": 0.1, "test": 0.2}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="summing to 1"):
        load_phase8_config(path)


def test_download_resumes_partial_and_writes_verified_sidecar(tmp_path, monkeypatch):
    import hashlib
    import urllib.request

    destination = tmp_path / "sentences_CC0.tar.bz2"
    destination.with_name(destination.name + ".part").write_bytes(b"part-")
    payload = b"rest"

    class FakeResponse:
        status = 206
        headers = {"Content-Length": "4"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def getcode(self): return 206
        def read(self, size=-1):
            nonlocal payload
            result, payload = payload[:size], payload[size:]
            return result

    seen = {}
    def fake_urlopen(request, timeout):
        seen["range"] = request.headers.get("Range")
        return FakeResponse()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    metadata = download_resumable("tatoeba_cc0", destination, expected_sha256=hashlib.sha256(b"part-rest").hexdigest())
    assert seen["range"] == "bytes=5-"
    assert destination.read_bytes() == b"part-rest"
    assert metadata["resumable_partial"] is False
    assert json.loads(destination.with_name(destination.name + ".provenance.json").read_text(encoding="utf-8"))["sha256"] == metadata["sha256"]


def test_benchmark_and_streamed_quality_scores(tmp_path):
    path = tmp_path / "quality.jsonl"
    rows = [{"text": "hello"}, {"text": "hello "}, {"text": "مرحبا"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    quality = corpus_statistics(path)
    assert quality["record_count_total"] == 3
    assert quality["quality"]["records"] == 3
    assert quality["quality"]["mean_record_score"] < 100
    measured = benchmark_jsonl(path, checkpoints=(1, 2, 5))
    assert measured["records_read"] == 3 and measured["unreached_checkpoints"] == [5]
    assert measured["data_origin"] == "actual imported corpus file"
    source = tmp_path / "sample.txt"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    source_benchmark = benchmark_source_records(source, checkpoints=(1, 3, 5))
    assert source_benchmark["records_read"] == 3
    assert source_benchmark["unreached_checkpoints"] == [5]

