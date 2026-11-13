import json
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.configuration import load_config
from evaluation.benchmark import benchmark_model, estimate_model_resources
from evaluation.engine import evaluate
from evaluation.reporting import build_report, write_reports
from evaluation.text_quality import evaluate_tokenizer
from model.factory import create_model
from tokenizer.byte_tokenizer import ByteTokenizer


class FixedModel(nn.Module):
    def __init__(self, vocab_size=260, bad=None):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.vocab_size, self.bad = vocab_size, bad
        self.config = type("Config", (), {"vocab_size": vocab_size, "context_length": 16,
                                           "d_model": 8, "n_heads": 2, "n_layers": 1})()

    def forward(self, idx, targets=None):
        logits = self.anchor * torch.ones((*idx.shape, self.vocab_size), device=idx.device)
        if self.bad == "nan":
            logits = logits * torch.tensor(float("nan"))
        elif self.bad == "inf":
            logits = logits * torch.tensor(float("inf"))
        if self.bad == "high_loss":
            logits = torch.full_like(logits, -1e6)
            logits[..., 0] = 0
        return logits, None


def make_batch(input_ids, target_ids):
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def test_average_loss_is_token_weighted_across_unequal_batches():
    model = FixedModel()
    # One-token batch and three-token batch; all logits are uniform.
    result = evaluate(model, [make_batch([[0]], [[0]]), make_batch([[1, 2, 3]], [[1, 2, 3]])])
    expected = math.log(260)
    assert result.status == "ok" and result.valid_tokens == 4 and result.batches == 2
    assert result.average_loss == pytest.approx(expected)
    assert result.perplexity == pytest.approx(260.0)


def test_padding_is_masked_but_eos_bos_unk_remain_valid_targets():
    tokenizer = ByteTokenizer()
    model = FixedModel()
    target = [[tokenizer.pad_id, tokenizer.eos_id, tokenizer.bos_id, tokenizer.unk_id, 17]]
    result = evaluate(model, [make_batch([[1, 1, 1, 1, 1]], target)], tokenizer=tokenizer)
    assert result.valid_tokens == 4
    assert result.padding_tokens_ignored == 1
    assert result.average_loss == pytest.approx(math.log(260))


def test_perplexity_returns_none_when_exponent_would_overflow():
    result = evaluate(FixedModel(bad="high_loss"), [make_batch([[1]], [[1]])])
    assert result.status == "ok"
    assert result.average_loss > 700
    assert result.perplexity is None
    assert result.warnings


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_nonfinite_logits_return_clear_error(bad):
    result = evaluate(FixedModel(bad=bad), [make_batch([[1]], [[1]])])
    assert result.status == "error" and result.average_loss is None
    assert "FloatingPointError" in result.error


def test_empty_loader_and_empty_valid_targets_are_handled():
    empty = evaluate(FixedModel(), [])
    assert empty.status == "empty" and empty.perplexity is None
    tok = ByteTokenizer()
    padded = evaluate(FixedModel(), [make_batch([[0, 0]], [[tok.pad_id, tok.pad_id]])], tokenizer=tok)
    assert padded.status == "empty_targets" and padded.valid_tokens == 0


def test_validation_and_test_loaders_are_independent_and_preserve_model_state():
    model = FixedModel().train()
    original = model.anchor.detach().clone()
    model.anchor.grad = torch.tensor(7.0)
    grad_before = model.anchor.grad.clone()
    validation = evaluate(model, [make_batch([[1]], [[1]])], split="validation")
    test_set = evaluate(model, [make_batch([[2, 3]], [[2, 3]])], split="test")
    assert validation.split == "validation" and test_set.split == "test"
    assert validation.valid_tokens == 1 and test_set.valid_tokens == 2
    assert model.training and torch.equal(model.anchor, original)
    assert torch.equal(model.anchor.grad, grad_before)


def test_invalid_batch_is_reported_without_partial_loss_claim():
    result = evaluate(FixedModel(), [(torch.ones((1, 2)), torch.ones((1, 1), dtype=torch.long))])
    assert result.status == "error" and result.average_loss is None
    assert "matching" in result.error


@pytest.mark.parametrize("name,text", [
    ("Arabic", "اللغة العربية سليمة؟"),
    ("English", "Text, numbers 123 and punctuation!"),
    ("French", "été, cœur, français et ç"),
    ("Mixed", "عربي English français 42 🙂"),
    ("Unicode", "e\u0301 漢字 🚀\x00"),
    ("Empty", ""),
])
def test_tokenizer_multilingual_round_trip(name, text):
    result = evaluate_tokenizer(ByteTokenizer())
    assert result["checks"][name.lower() if name != "Mixed" else "mixed"]["round_trip"]
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text), errors="strict") == text


def test_tokenizer_quality_checks_lengths_ids_and_incomplete_utf8():
    result = evaluate_tokenizer(ByteTokenizer(), context_length=128)
    assert result["checks"]["long_text"]["exceeds_context"]
    assert result["checks"]["long_text"]["round_trip"]
    assert result["checks"]["incomplete_utf8"]["strict_decode_rejected"]
    assert result["checks"]["empty"]["token_count"] == 0
    assert result["checks"]["mixed"]["valid_ids"]


def test_generation_mechanics_short_prompt_cpu():
    settings = load_config()
    model, _ = create_model(settings)
    rng_before = torch.random.get_rng_state().clone()
    result = evaluate_tokenizer(ByteTokenizer(), model=model, context_length=settings.context_length, max_new_tokens=2)
    generation = result["generation"]
    assert generation["prompt_tokens"] == 2
    assert generation["generated_tokens"] <= 2 and generation["ids_valid"]
    assert generation["model_training_state_preserved"]
    assert generation["over_context_input_rejected"]
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_benchmark_metrics_and_parameter_count_are_bounded_cpu():
    settings = load_config()
    model, _ = create_model(settings)
    rng_before = torch.random.get_rng_state().clone()
    benchmark = benchmark_model(model, ByteTokenizer(), iterations=1, warmup=0, max_seconds=30,
                               model_factory=lambda: create_model(settings)[0])
    assert benchmark["device"] == "cpu" and benchmark["status"] == "ok"
    assert benchmark["total_parameters_measured"] == sum(p.numel() for p in model.parameters())
    assert benchmark["average_evaluation_batch_latency_seconds"] > 0
    assert benchmark["average_generation_latency_seconds"] > 0
    assert benchmark["elapsed_seconds_measured"] < 30
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_resource_estimates_are_calculated_without_building_models():
    settings = load_config()
    estimates = estimate_model_resources(settings)
    profiles = {profile["name"]: profile for profile in estimates["profiles"]}
    assert profiles["current"]["parameters_calculated"] == 843008
    assert profiles["current"]["fp32_weights_mib_theoretical"] == pytest.approx(3.21582, rel=1e-5)
    assert profiles["tiny"]["fp16_weights_mib_theoretical"] < profiles["current"]["fp16_weights_mib_theoretical"]
    assert "Excludes activations" in profiles["current"]["note"]


def test_json_and_markdown_reports_are_valid_and_non_overwriting(tmp_path):
    settings = load_config()
    model, _ = create_model(settings)
    result = evaluate(model, [make_batch([[1]], [[1]])])
    report = build_report(model=model, settings=settings, evaluation=result.to_dict(),
                          benchmark={"average_generation_latency_seconds": float("nan")},
                          tokenizer_results={"status": "ok"}, resource_estimates=estimate_model_resources(settings))
    paths = write_reports(report, tmp_path)
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    assert payload["benchmark"]["average_generation_latency_seconds"] is None
    assert "Average loss" in open(paths["markdown"], encoding="utf-8").read()
    with pytest.raises(FileExistsError):
        write_reports(report, tmp_path)
