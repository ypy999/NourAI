import json

import pytest
import torch

from model.transformer import GPT, GPTConfig, Block, causal_mask
from tokenizer.byte_tokenizer import ByteTokenizer


def tiny_config():
    return GPTConfig(vocab_size=260, context_length=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)


def test_config_and_default_json_dimensions_agree():
    with open("configs/default.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tok = ByteTokenizer()
    assert cfg["vocab_size"] == tok.vocab_size == GPTConfig().vocab_size
    assert cfg["d_model"] % cfg["n_heads"] == 0


@pytest.mark.parametrize("kwargs", [
    {"vocab_size": 0}, {"context_length": 0}, {"d_model": 30, "n_heads": 4},
    {"n_layers": 0}, {"dropout": 1.0},
])
def test_reject_invalid_config(kwargs):
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


def test_causal_mask_blocks_future_only():
    m = causal_mask(4)
    assert m.tolist() == [[False, True, True, True], [False, False, True, True],
                          [False, False, False, True], [False, False, False, False]]


def test_future_tokens_do_not_change_previous_block_outputs():
    torch.manual_seed(1)
    block = Block(tiny_config()).eval()
    x1 = torch.randn(1, 6, 32)
    x2 = x1.clone()
    x2[:, 4:, :] += 100
    with torch.no_grad():
        y1, y2 = block(x1), block(x2)
    torch.testing.assert_close(y1[:, :4], y2[:, :4], rtol=0, atol=1e-6)


def test_forward_shapes_loss_and_backward_gradients():
    model = GPT(tiny_config())
    x = torch.randint(0, 260, (2, 8))
    y = torch.randint(0, 260, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 260)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for parameter in (model.token.weight, model.blocks[0].attn.in_proj_weight, model.blocks[0].ff[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


@pytest.mark.parametrize("x", [torch.empty((1, 0), dtype=torch.long), torch.zeros((1, 17), dtype=torch.long), torch.full((1, 2), 260)])
def test_forward_rejects_invalid_tokens_or_length(x):
    with pytest.raises(ValueError):
        GPT(tiny_config())(x)


def test_forward_rejects_target_shape_mismatch():
    with pytest.raises(ValueError, match="same"):
        GPT(tiny_config())(torch.ones((1, 3), dtype=torch.long), torch.ones((1, 2), dtype=torch.long))


def test_generation_eos_stops_and_restores_training_mode():
    class EosModel(GPT):
        def forward(self, idx, targets=None):
            logits = torch.zeros(idx.size(0), idx.size(1), self.config.vocab_size, device=idx.device)
            logits[..., ByteTokenizer.eos_id] = 100
            return logits, None

    model = EosModel(tiny_config()).train()
    out = model.generate(torch.tensor([[1, 2]]), max_new_tokens=5, temperature=1.0, top_k=1, eos_id=258)
    assert out.shape == (1, 3)
    assert out[0, -1].item() == 258
    assert model.training


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1}, {"temperature": 0}, {"top_k": -1}, {"eos_id": 260},
])
def test_generation_rejects_invalid_options(kwargs):
    model = GPT(tiny_config())
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1]]), **kwargs)


def test_generation_rejects_empty_prompt():
    with pytest.raises(ValueError):
        GPT(tiny_config()).generate(torch.empty((1, 0), dtype=torch.long))


@pytest.mark.parametrize("text", ["مرحبا بالعالم", "Hello, world!", "Bonjour, ça va ?", "🙂🚀", "a\x00b", "العربية English français"])
def test_tokenizer_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text


def test_tokenizer_special_and_invalid_ids():
    tok = ByteTokenizer()
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.vocab_size) == (256, 257, 258, 259, 260)
    assert tok.decode([tok.bos_id, 65, tok.eos_id]) == "A"
    with pytest.raises(ValueError):
        tok.decode([260])
    with pytest.raises(TypeError):
        tok.decode([1.5])


def test_tokenizer_invalid_utf8_policy():
    tok = ByteTokenizer()
    assert tok.decode([0xD8], errors="replace") == "�"
    with pytest.raises(UnicodeDecodeError):
        tok.decode([0xD8], errors="strict")


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    model = GPT(tiny_config()).eval()
    ids = torch.randint(0, 260, (1, 5))
    with torch.no_grad():
        expected, _ = model(ids)
    path = tmp_path / "tiny.pt"
    model.save_checkpoint(path)
    restored = GPT.load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        actual, _ = restored(ids)
    torch.testing.assert_close(actual, expected)
    assert restored.config == model.config
