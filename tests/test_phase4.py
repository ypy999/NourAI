import json
from pathlib import Path

import pytest
import torch

from configs.configuration import ProjectConfig, load_config
from data.processor import DataPipeline, DataPipelineError
from datasets.loader import create_dataloaders
from model.factory import create_model
from tokenizer.byte_tokenizer import ByteTokenizer


def cfg(**changes):
    base = dict(vocab_size=260, context_length=8, d_model=16, n_heads=4, n_layers=1,
                dropout=0.0, batch_size=2, learning_rate=0.001, device="cpu",
                validation_fraction=0.2, random_seed=3, max_document_chars=10000, io_batch_lines=16)
    base.update(changes)
    return ProjectConfig(**base).validate(ByteTokenizer())


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_valid_config_and_defaults(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"d_model": 24, "n_heads": 3})
    loaded = load_config(path)
    assert loaded.d_model == 24 and loaded.n_heads == 3
    assert loaded.n_layers == 4 and loaded.vocab_size == ByteTokenizer.vocab_size


@pytest.mark.parametrize("contents", ["{bad", "[]", '{"batch_size": true}', '{"unexpected": 1}', '{"vocab_size": 259}'])
def test_reject_invalid_or_incompatible_config(tmp_path, contents):
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_reject_dimension_and_training_configuration_conflicts():
    with pytest.raises(ValueError, match="divisible"):
        cfg(d_model=15, n_heads=4)
    with pytest.raises(ValueError, match="validation_fraction"):
        cfg(validation_fraction=1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        cfg(learning_rate=0)


def test_model_factory_loads_json_and_changes_architecture(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"d_model": 24, "n_heads": 3, "n_layers": 2, "context_length": 16})
    model, settings = create_model(path)
    assert settings.d_model == model.config.d_model == 24
    assert len(model.blocks) == 2
    assert model.blocks[0].attn.num_heads == 3
    assert model.pos.num_embeddings == 16


def test_factory_rejects_tokenizer_vocabulary_mismatch(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"vocab_size": 259})
    with pytest.raises(ValueError, match="tokenizer"):
        create_model(path)


def make_sources(root, count=400):
    txt = root / "a.txt"
    jsonl = root / "b.jsonl"
    records = [f"وثيقة عربية English français numéro {i} avec assez de texte pour le contexte." for i in range(count)]
    txt.write_text("\n".join(records[:count // 2]) + "\n", encoding="utf-8")
    with jsonl.open("w", encoding="utf-8") as f:
        for text in records[count // 2:]:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"text": records[0]}, ensure_ascii=False) + "\n")
        f.write('{broken json}\n')
        f.write('{"content": "   "}\n')
    return txt, jsonl, records


def test_txt_jsonl_clean_deduplicate_split_and_arabic(tmp_path):
    txt, jsonl, records = make_sources(tmp_path)
    out = tmp_path / "processed"
    stats = DataPipeline(cfg()).process([txt, jsonl], out)
    train = [json.loads(line)["text"] for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [json.loads(line)["text"] for line in (out / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(records)
    assert stats.duplicate_records == 1 and stats.invalid_records == 1 and stats.empty_records == 1
    assert stats.train_records + stats.validation_records == len(records)
    assert any("وثيقة عربية" in text for text in train + val)
    assert (txt.exists() and jsonl.exists())


def test_pipeline_counts_corrupt_utf8_and_oversized_lines(tmp_path):
    source = tmp_path / "broken.txt"
    source.write_bytes(("\n".join(f"valid short row {i}" for i in range(100)) + "\n").encode("utf-8") + b"\xff\xfe\n" + b"x" * 120 + b"\n")
    out = tmp_path / "out"
    stats = DataPipeline(cfg(max_document_chars=50, validation_fraction=0.5)).process([source], out)
    assert stats.invalid_records == 1 and stats.oversized_records == 1
    # Processing leaves the original source unchanged.
    assert source.read_bytes().startswith(b"valid short row 0")


def test_empty_inputs_and_existing_outputs_are_reported(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text(" \n\t\n", encoding="utf-8")
    with pytest.raises(DataPipelineError, match="no valid"):
        DataPipeline(cfg()).process([empty], tmp_path / "empty-out")
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "train.jsonl").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        DataPipeline(cfg()).process([empty], out)
    assert (out / "train.jsonl").read_text(encoding="utf-8") == "preserve"


def test_dataloader_shapes_types_and_fixed_sequence_length(tmp_path):
    txt, jsonl, _ = make_sources(tmp_path, count=80)
    processed = tmp_path / "processed"
    settings = cfg()
    DataPipeline(settings).process([txt, jsonl], processed)
    train, validation = create_dataloaders(processed, settings)
    x, y = next(iter(train))
    assert x.shape == y.shape == (settings.batch_size, settings.context_length)
    assert x.dtype == y.dtype == torch.int64
    assert torch.all((x >= 0) & (x < settings.vocab_size))
    assert len(validation.dataset) > 0


def test_dataset_reports_short_or_corrupt_files(tmp_path):
    from datasets.dataset import TokenSequenceDataset
    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps({"text": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least"):
        TokenSequenceDataset(short, cfg())
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        TokenSequenceDataset(corrupt, cfg())


def test_large_file_is_processed_and_records_are_accounted(tmp_path):
    source = tmp_path / "large.txt"
    source.write_text("\n".join(f"record {i} long enough for token sequence" for i in range(2000)), encoding="utf-8")
    stats = DataPipeline(cfg(validation_fraction=0.25)).process([source], tmp_path / "processed")
    assert stats.accepted_records == 2000
    assert stats.train_records + stats.validation_records == 2000
