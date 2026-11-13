"""Phase 7 tests are added incrementally as dataset components are implemented."""
import json
import sqlite3
import tracemalloc

import pytest

from configs.configuration import ProjectConfig, load_config
from data.processor import DataPipeline, DataPipelineStats, clean_text, estimate_sources
from data.deduplication import DiskDeduplicator, normalize_duplicate_key
from data.language_analysis import analyze_records, classify_language
from data.quality import RECORD_WEIGHTS, score_record, summarize_quality
from data.splitting import split_records
from data.tokenizer_analysis import BYTE_TOKENIZER_VERSION, analyze_tokenizer_records
from data.manifest import build_manifest, training_readiness, write_manifest
from data.reporting import build_dataset_report, memory_snapshot, write_dataset_reports
from data.splitting import split_sources
from tokenizer.byte_tokenizer import ByteTokenizer


def test_cleaning_defaults_are_centralized_and_validated(tmp_path):
    config = load_config()
    assert config.cleaning_unicode_normalization == "NFC"
    assert config.cleaning_remove_control_chars is True
    assert config.cleaning_collapse_whitespace is True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"cleaning_min_chars": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="cleaning_min_chars"):
        load_config(path)
    with pytest.raises(ValueError, match="normalization"):
        ProjectConfig(cleaning_unicode_normalization="BAD").validate(ByteTokenizer())


def test_clean_text_preserves_punctuation_arabic_diacritics_and_joiners():
    text = "  إِنَّها، جميلة!  \r\n\r\n👨‍👩‍👧‍👦  "
    cleaned = clean_text(text, preserve_paragraphs=True)
    assert cleaned == "إِنَّها، جميلة!\n\n👨‍👩‍👧‍👦"
    assert "إِنَّها، جميلة!" in cleaned


def test_clean_text_configurable_unicode_and_whitespace():
    decomposed = "e\u0301  test\tspacing"
    assert clean_text(decomposed) == "é test spacing"
    assert clean_text(decomposed, unicode_normalization="NFD") == "e\u0301 test spacing"
    assert clean_text("a\x00b") == "ab"
    assert clean_text("a\x00b", remove_control_chars=False) == "a\x00b"


def test_txt_and_jsonl_import_streams_source_metadata_and_keeps_inputs(tmp_path):
    txt = tmp_path / "arabic.txt"
    jsonl = tmp_path / "records.jsonl"
    txt.write_text("".join(f"مرحباً بالعالم، هذا نص طويل enough number {index}\n" for index in range(20)), encoding="utf-8")
    jsonl.write_text(json.dumps({"content": "Bonjour, ceci est une phrase assez longue."}, ensure_ascii=False) + "\n", encoding="utf-8")
    original_txt = txt.read_bytes()
    original_jsonl = jsonl.read_bytes()
    output = tmp_path / "processed"
    config_path = tmp_path / "import_config.json"
    config_path.write_text(json.dumps({"validation_fraction": 0.2}), encoding="utf-8")
    stats = DataPipeline(config_path).process([txt, jsonl], output)
    rows = [json.loads(line) for name in ("train.jsonl", "validation.jsonl")
            for line in (output / name).read_text(encoding="utf-8").splitlines()]
    assert stats.accepted_records == 21
    assert stats.processing_seconds >= 0
    assert stats.source_estimate["estimated_input_bytes"] == len(original_txt) + len(original_jsonl)
    assert len(rows) == 21
    assert all(row["source"] in (str(txt.resolve()), str(jsonl.resolve())) for row in rows)
    assert all(isinstance(row["line_number"], int) and row["line_number"] > 0 for row in rows)
    assert (txt.read_bytes(), jsonl.read_bytes()) == (original_txt, original_jsonl)


def test_invalid_utf8_and_malformed_json_records_are_reported(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b'{"text":"valid enough record"}\n\xff\n{broken}\n')
    counters = DataPipelineStats()
    records = list(DataPipeline()._iter_records([source], counters))
    assert [record["text"] for record in records] == ["valid enough record"]
    assert counters.invalid_records == 2


def test_unsupported_nested_files_are_reported(tmp_path):
    (tmp_path / "ok.txt").write_text("valid record long enough\n", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("must not be read", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.bin").write_bytes(b"\x00\x01")
    from data.processor import DataPipelineStats
    stats = DataPipelineStats()
    records = list(DataPipeline()._iter_records([tmp_path], stats))
    assert stats.unsupported_files == 2
    assert any("ignored.csv" in error for error in stats.errors)
    assert len(records) == 1


def test_short_record_policy_is_configurable(tmp_path):
    source = tmp_path / "short.txt"
    source.write_text("short\n" + "".join(f"long enough unique record {index}\n" for index in range(20)), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"cleaning_min_chars": 10, "cleaning_reject_short": True}), encoding="utf-8")
    pipeline = DataPipeline(config_path)
    stats = pipeline.process([source], tmp_path / "out")
    assert stats.short_records == 1
    assert stats.accepted_records == 20


def test_disk_deduplication_distinguishes_exact_and_normalized_hash_duplicates():
    connection = sqlite3.connect(":memory:")
    try:
        dedup = DiskDeduplicator(connection)
        first = {"text": "Hello, world!", "source": "a.txt", "line_number": 1}
        assert dedup.add_if_unique(first) is None
        assert dedup.add_if_unique({**first, "source": "b.txt", "line_number": 4}) == "exact"
        assert dedup.add_if_unique({**first, "text": "  hello,   WORLD!  ", "source": "b.txt"}) == "normalized"
        saved = connection.execute("SELECT source, line_number FROM seen_records").fetchone()
        assert saved == ("a.txt", 1)
        assert normalize_duplicate_key("  ÉTÉ  \t test ") == normalize_duplicate_key("été test")
    finally:
        connection.close()


def test_language_classification_and_statistics_are_local_and_heuristic():
    arabic = "هذه جملة باللغة العربية"
    english = "Hello, this is the English text"
    french = "Bonjour, merci pour le français"
    mixed = "مرحباً hello world"
    assert classify_language(arabic) == "arabic"
    assert classify_language(english) == "english"
    assert classify_language(french) == "french"
    assert classify_language(mixed) == "mixed"
    assert classify_language("123 🙂") == "unclassified"
    stats = analyze_records([arabic, english, french, mixed, "123 🙂"])
    assert stats.records == 5
    assert stats.records_by_language == {"arabic": 1, "english": 1, "french": 1, "mixed": 1, "unclassified": 1}
    assert stats.total_characters == sum(map(len, [arabic, english, french, mixed, "123 🙂"]))
    assert stats.characters_by_language["french"] > 0
    assert stats.characters_by_language["english"] > 0
    assert stats.average_record_characters > 0
    assert sum(stats.percentages.values()) == pytest.approx(100.0)


def _multilingual_records():
    records = []
    for index in range(20):
        records.append({"text": f"هذه جملة عربية فريدة للاختبار رقم {index}", "source": "ar.jsonl", "line_number": index + 1})
        records.append({"text": f"Hello this is English text number {index}", "source": "en.jsonl", "line_number": index + 1})
        records.append({"text": f"Bonjour merci pour le français numéro {index}", "source": "fr.jsonl", "line_number": index + 1})
    return records


def _read_split(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_quality_score_formula_is_bounded_and_documented():
    assert sum(RECORD_WEIGHTS.values()) == pytest.approx(1.0)
    perfect = score_record("A complete English text.")
    duplicate = score_record("A complete English text.", duplicate=True)
    short = score_record("x", min_chars=5)
    assert perfect["score"] == 100.0
    assert duplicate["score"] == 80.0
    assert short["score"] < perfect["score"]
    summary = summarize_quality([perfect | {"language": "arabic"}, perfect | {"language": "english"},
                                 perfect | {"language": "french"}])
    assert summary.language_balance_score == pytest.approx(100.0)
    assert summary.overall_dataset_score == pytest.approx(100.0)
    assert "0.15" in summary.score_formula


def test_quality_summary_penalizes_language_imbalance_but_not_as_semantic_quality():
    scores = [score_record("العربية") | {"language": "arabic"} for _ in range(3)]
    summary = summarize_quality(scores)
    assert summary.mean_record_score == 100.0
    assert summary.language_balance_score == 0.0
    assert summary.overall_dataset_score == 90.0


def test_three_way_split_prevents_exact_and_normalized_leakage_and_is_reproducible(tmp_path):
    rows = _multilingual_records()
    rows.extend([rows[0], {**rows[1], "text": "  " + rows[1]["text"].upper() + "  ", "source": "duplicate-source"}])
    first = split_records(iter(rows), tmp_path / "run-a", seed=77)
    second = split_records(iter(rows), tmp_path / "run-b", seed=77)
    assert first.duplicate_records == 2
    assert first.unique_records == 60
    assert all(first.split_counts[name] > 0 for name in ("train", "validation", "test"))
    sets = {name: {normalize_duplicate_key(item["text"]) for item in _read_split(tmp_path / "run-a" / f"{name}.jsonl")}
            for name in ("train", "validation", "test")}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert first.split_counts == second.split_counts
    for name in ("train", "validation", "test"):
        assert (tmp_path / "run-a" / f"{name}.jsonl").read_bytes() == (tmp_path / "run-b" / f"{name}.jsonl").read_bytes()
    assert first.language_counts == second.language_counts


def test_small_dataset_split_is_reproducible_and_reports_empty_requested_splits(tmp_path):
    stats = split_records([{"text": "short data", "source": "known.txt"}, {"text": "  \n "}], tmp_path / "small", seed=4)
    assert stats.empty_records == 1
    assert stats.split_counts == {"train": 1, "validation": 0, "test": 0}
    assert stats.warnings
    train = _read_split(tmp_path / "small" / "train.jsonl")
    assert train[0]["source"] == "known.txt"
    assert (tmp_path / "small" / "validation.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "small" / "test.jsonl").read_text(encoding="utf-8") == ""


def test_split_ratios_and_seed_are_validated(tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        split_records([], tmp_path / "bad", ratios={"train": 0.8, "validation": 0.1, "test": 0.2})
    with pytest.raises(ValueError, match="seed"):
        split_records([], tmp_path / "bad-seed", seed=-1)
    with pytest.raises(ValueError, match="ratio"):
        split_records([], tmp_path / "nan-ratio", ratios={"train": float("nan"), "validation": 0.0, "test": 0.0})


def test_tokenizer_analysis_multilingual_counts_and_context_overflow():
    samples = ["العربية", "Hello!", "Bonjour, merci pour le français", "mix العربية English", "🙂🚀"]
    result = analyze_tokenizer_records(iter(samples), ByteTokenizer(), context_length=8, chunk_characters=1)
    expected = sum(len(ByteTokenizer().encode(text, add_eos=True)) for text in samples)
    assert result["records"] == len(samples)
    assert result["tokens"] == expected
    assert result["token_cache_bytes_uint16_estimate"] == expected * 2
    assert result["unique_token_ids"] <= 260
    assert result["records_over_context"] > 0
    assert set(result["by_language"]) >= {"arabic", "english", "french", "mixed"}
    assert result["tokenizer_version"] == BYTE_TOKENIZER_VERSION


def test_tokenizer_analysis_empty_and_invalid_records():
    empty = analyze_tokenizer_records([], context_length=16)
    assert empty["records"] == 0 and empty["tokens_per_character"] is None
    with pytest.raises(ValueError, match="string 'text'"):
        analyze_tokenizer_records([{"not_text": "bad"}])


def test_manifest_does_not_invent_provenance_and_blocks_unknown_license(tmp_path):
    blocked = build_manifest(dataset_name="test-fixture", dataset_version="0.1", source="synthetic test records",
                             license=None, pipeline_statistics={"accepted_records": 2})
    assert blocked["license"] is None
    assert blocked["collection_date"] is None
    assert blocked["training_eligible"] is False
    assert "license is missing or unverified" in blocked["training_blockers"]
    assert training_readiness(blocked)["eligible"] is False
    path = write_manifest(blocked, tmp_path / "dataset.manifest.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["license"] is None and loaded["training_eligible"] is False
    with pytest.raises(FileExistsError):
        write_manifest(blocked, path)


def test_manifest_accepts_only_explicit_source_and_license_for_readiness(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")
    manifest = build_manifest(
        dataset_name="licensed-example", dataset_version="1.0", source={"name": "User supplied source"},
        license="CC-BY-4.0 as supplied by user", collection_date="2025-01-15",
        pipeline_statistics={"accepted_records": 6, "output_characters": 120},
        split_statistics={"train": 4, "validation": 1, "test": 1},
        processed_files=[data_file],
    )
    assert manifest["training_eligible"] is True
    assert manifest["file_size_bytes"] == data_file.stat().st_size
    assert manifest["tokenizer_version"] == BYTE_TOKENIZER_VERSION
    assert training_readiness(manifest) == {"eligible": True, "blockers": []}


def test_large_generator_split_is_disk_backed_and_does_not_change_source(tmp_path):
    def rows():
        for index in range(4000):
            yield {"text": f"Hello this is a unique English record for streaming number {index}",
                   "source": "generated-test-fixture", "line_number": index + 1}
    tracemalloc.start()
    try:
        stats = split_records(rows(), tmp_path / "large", seed=19)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert stats.input_records == stats.unique_records == 4000
    assert sum(stats.split_counts.values()) == 4000
    assert peak < 64 * 1024 * 1024


def test_source_size_estimate_uses_file_metadata_without_reading_contents(tmp_path):
    text = tmp_path / "small.txt"
    unsupported = tmp_path / "skip.csv"
    text.write_bytes(b"x" * 8192)
    unsupported.write_bytes(b"y" * 16384)
    estimate = estimate_sources([tmp_path])
    assert estimate["supported_files"] == 1
    assert estimate["unsupported_files"] == 1
    assert estimate["estimated_input_bytes"] == 8192


def test_split_sources_streams_txt_and_keeps_raw_bytes_unchanged(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("".join(f"Hello this is an English source record number {i}\n" for i in range(60)), encoding="utf-8")
    original = source.read_bytes()
    stats = split_sources([source], tmp_path / "source-splits", seed=123)
    assert stats.source_import_stats["records_seen"] == 60
    assert stats.unique_records == 60
    assert source.read_bytes() == original
    assert sum(stats.split_counts.values()) == 60


@pytest.mark.parametrize("interrupt_at", [1, 5])
def test_split_checkpoint_resumes_from_committed_batch_and_validates_fingerprint(tmp_path, interrupt_at):
    records = [{"text": f"Hello this is checkpoint record {index}", "source": "test", "line_number": index + 1}
               for index in range(12)]
    checkpoint = tmp_path / "checkpoint.sqlite"
    output = tmp_path / "resumed-splits"

    def interrupted():
        for index, record in enumerate(records):
            if index == interrupt_at:
                raise RuntimeError("simulated interruption")
            yield record

    with pytest.raises(RuntimeError, match="interruption"):
        split_records(interrupted(), output, seed=12, checkpoint_path=checkpoint,
                      source_fingerprint="fixture-v1", checkpoint_batch_size=3)
    assert checkpoint.exists()
    resumed = split_records(iter(records), output, seed=12, checkpoint_path=checkpoint,
                            source_fingerprint="fixture-v1", resume=True, checkpoint_batch_size=3)
    assert resumed.input_records == resumed.unique_records == 12
    assert sum(resumed.split_counts.values()) == 12
    with pytest.raises(ValueError, match="fingerprint"):
        split_records(iter(records), tmp_path / "wrong", seed=12, checkpoint_path=checkpoint,
                      source_fingerprint="changed-fixture", resume=True)


def test_dataset_report_writes_json_markdown_and_memory_snapshot_shape(tmp_path):
    report = build_dataset_report(dataset_overview={"name": "no-corpus", "record_count": 0, "corpus_present": False},
                                  warnings=["No user corpus was supplied"], test_results={"passed": 1, "failed": 0})
    assert set(memory_snapshot()) == {"total_ram_bytes", "available_ram_bytes", "process_working_set_bytes", "process_peak_working_set_bytes"}
    paths = write_dataset_reports(report, tmp_path, stem="phase7")
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    markdown = open(paths["markdown"], encoding="utf-8").read()
    assert payload["dataset_overview"]["corpus_present"] is False
    assert "No user corpus was supplied" in markdown
    with pytest.raises(FileExistsError):
        write_dataset_reports(report, tmp_path, stem="phase7")
