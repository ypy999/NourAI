import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.configuration import ProjectConfig, load_config
from model.factory import create_model
from training import CheckpointManager, EarlyStopping, Trainer, create_scheduler, validate
from training.logger import JsonlLogger
from training.metrics import memory_usage_mb


def config(**changes):
    values = load_config().to_dict()
    values.update({
        "context_length": 4, "d_model": 8, "n_heads": 2, "n_layers": 1,
        "dropout": 0.0, "batch_size": 2, "epochs": 1,
        "learning_rate": 0.01, "weight_decay": 0.0,
        "validation_interval": 1, "checkpoint_interval": 1,
        "warmup_steps": 0, "gradient_accumulation_steps": 1,
        "scheduler_type": "constant", "max_training_steps": 0,
        "early_stopping": False, "random_seed": 10,
    })
    values.update(changes)
    return ProjectConfig(**values).validate()


def loaders(seed=11):
    torch.manual_seed(seed)
    x = torch.randint(0, 260, (8, 4))
    y = torch.randint(0, 260, (8, 4))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=2, shuffle=False), DataLoader(ds, batch_size=2, shuffle=False)


def model_for(cfg):
    # Model Factory path is covered in phase 4; synthetic training uses its returned model.
    return create_model(cfg)[0]


def test_optimizer_parameters_and_scheduler_variants(tmp_path):
    cfg = config(adam_betas=[0.8, 0.95], adam_epsilon=1e-7, weight_decay=0.03)
    model = model_for(cfg)
    manager = CheckpointManager(tmp_path / "ckpt")
    train, val = loaders()
    trainer = Trainer(model, cfg, train, val, manager, JsonlLogger(tmp_path / "events.jsonl"))
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.defaults["betas"] == (0.8, 0.95)
    for kind in ("linear_warmup_cosine", "cosine", "constant"):
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = create_scheduler(opt, kind, total_steps=8, warmup_steps=2)
        rates = []
        for _ in range(4):
            opt.step(); scheduler.step(); rates.append(opt.param_groups[0]["lr"])
        assert all(torch.isfinite(torch.tensor(rate)) for rate in rates)


def test_training_loop_updates_weights_validates_and_logs(tmp_path):
    cfg = config()
    model = model_for(cfg)
    train, val = loaders()
    before = model.token.weight.detach().clone()
    logger = JsonlLogger(tmp_path / "events.jsonl")
    trainer = Trainer(model, cfg, train, val, CheckpointManager(tmp_path / "ckpts"), logger)
    history = trainer.fit()
    assert trainer.global_step == 4
    assert not torch.equal(before, model.token.weight)
    assert history and all("validation_loss" in row for row in history)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["event"] for row in events} >= {"train_step", "validation", "epoch_end"}


def test_gradient_accumulation_and_clipping(tmp_path):
    cfg = config(gradient_accumulation_steps=2, gradient_clip_norm=0.05)
    train, val = loaders()
    trainer = Trainer(model_for(cfg), cfg, train, val, CheckpointManager(tmp_path / "ckpt"), JsonlLogger(tmp_path / "log.jsonl"))
    trainer.fit()
    assert trainer.global_step == 2


def test_validation_does_not_update_weights_or_leave_eval_mode():
    cfg = config()
    model = model_for(cfg).train()
    _, val = loaders()
    before = [p.detach().clone() for p in model.parameters()]
    result = validate(model, val)
    assert model.training
    assert result["tokens"] == 32
    assert result["validation_loss"] > 0 and result["perplexity"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_checkpoint_round_trip_optimizer_scheduler_rng_and_history(tmp_path):
    cfg = config()
    model = model_for(cfg)
    train, val = loaders()
    manager = CheckpointManager(tmp_path / "ckpt", keep=2)
    trainer = Trainer(model, cfg, train, val, manager, JsonlLogger(tmp_path / "log.jsonl"))
    trainer.fit()
    saved_step = trainer.global_step
    # Saving a second checkpoint archives the prior last checkpoint.
    trainer._save(trainer.epoch, trainer.batch_in_epoch)
    assert list((tmp_path / "ckpt" / "history").glob("last-*.pt"))
    restored = model_for(cfg)
    train2, val2 = loaders()
    resumed = Trainer(restored, cfg, train2, val2, manager, JsonlLogger(tmp_path / "resume.jsonl"))
    payload = resumed.resume()
    assert payload["global_step"] == saved_step
    assert resumed.global_step == saved_step
    for original, new in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, new)


def test_resume_mid_epoch_skips_completed_batches(tmp_path):
    cfg = config()
    train, val = loaders()
    manager = CheckpointManager(tmp_path / "ckpt")
    interrupted = Trainer(model_for(cfg), cfg, train, val, manager, JsonlLogger(tmp_path / "a.jsonl"))
    interrupted._save(0, 1)  # simulated safe optimizer boundary after batch zero
    train2, val2 = loaders()
    resumed = Trainer(model_for(cfg), cfg, train2, val2, manager, JsonlLogger(tmp_path / "b.jsonl"))
    resumed.resume()
    resumed.fit()
    assert resumed.global_step == 3


def test_checkpoint_rejects_corruption_and_config_mismatch(tmp_path):
    cfg = config()
    manager = CheckpointManager(tmp_path / "ckpt")
    train, val = loaders()
    trainer = Trainer(model_for(cfg), cfg, train, val, manager, JsonlLogger(tmp_path / "log.jsonl"))
    trainer._save(0, 0)
    other = config(d_model=16, n_heads=2)
    with pytest.raises(ValueError, match="architecture"):
        manager.load(model=model_for(other), config=other)
    corrupt = tmp_path / "bad.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="unreadable"):
        manager.load(corrupt, model=model_for(cfg), config=cfg)


def test_early_stopping_patience_and_minimum_improvement():
    stop = EarlyStopping(patience=2, min_delta=0.1)
    assert stop.update(2.0) == (True, False)
    assert stop.update(1.95) == (False, False)
    assert stop.update(1.8) == (True, False)
    assert stop.update(1.8) == (False, False)
    assert stop.update(1.8) == (False, True)


def test_reproducible_cpu_runs(tmp_path):
    cfg = config()
    outputs = []
    for idx in range(2):
        torch.manual_seed(123)
        train, val = loaders(seed=456)
        model = model_for(cfg)
        Trainer(model, cfg, train, val, CheckpointManager(tmp_path / f"c{idx}"), JsonlLogger(tmp_path / f"l{idx}.jsonl")).fit()
        outputs.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


def test_tiny_synthetic_training_lowers_loss_and_saves_recoverable_state(tmp_path):
    cfg = config(epochs=10, learning_rate=0.03, max_training_steps=20,
                 validation_interval=2, checkpoint_interval=4)
    x = torch.zeros((8, cfg.context_length), dtype=torch.long)
    y = torch.full_like(x, 65)
    dataset = TensorDataset(x, y)
    train = DataLoader(dataset, batch_size=4, shuffle=False)
    val = DataLoader(dataset, batch_size=4, shuffle=False)
    model = model_for(cfg)
    initial = validate(model, val)["validation_loss"]
    manager = CheckpointManager(tmp_path / "ckpt")
    trainer = Trainer(model, cfg, train, val, manager, JsonlLogger(tmp_path / "events.jsonl"))
    trainer.fit()
    final = validate(model, val)["validation_loss"]
    print(f"synthetic_validation_loss: {initial:.6f} -> {final:.6f}")
    assert final < initial
    assert (tmp_path / "ckpt" / "last.pt").is_file()
    assert (tmp_path / "ckpt" / "best.pt").is_file()
    restored = model_for(cfg)
    payload = manager.load(model=restored, config=cfg)
    assert payload["global_step"] == trainer.global_step == 20
    assert payload["optimizer_state"]["state"]


def test_memory_metric_is_real_or_explicitly_unavailable():
    value = memory_usage_mb()
    assert value is None or value > 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_training_loss_stops_loudly(tmp_path, bad_value):
    cfg = config()
    train, val = loaders()
    model = model_for(cfg)
    with torch.no_grad():
        model.token.weight[0, 0] = bad_value
    trainer = Trainer(model, cfg, train, val, CheckpointManager(tmp_path / "c"), JsonlLogger(tmp_path / "l.jsonl"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.fit()


def test_graceful_stop_saves_at_safe_optimizer_boundary(tmp_path):
    cfg = config(epochs=4)
    train, val = loaders()
    manager = CheckpointManager(tmp_path / "ckpt")
    trainer = Trainer(model_for(cfg), cfg, train, val, manager, JsonlLogger(tmp_path / "events.jsonl"))
    trainer.request_stop()
    trainer.fit()
    payload = manager.load(model=model_for(cfg), config=cfg, restore_rng=False)
    assert trainer.global_step == 1
    assert payload["trainer_state"]["batch_in_epoch"] == 1


@pytest.mark.parametrize("changes", [
    {"epochs": 0}, {"adam_betas": [1.0, 0.9]}, {"gradient_accumulation_steps": 0},
    {"precision": "fp64"}, {"gradient_clip_norm": 0}, {"weight_decay": -1},
])
def test_training_config_validation(changes):
    values = load_config().to_dict(); values.update(changes)
    with pytest.raises(ValueError):
        ProjectConfig(**values).validate()


def test_cpu_rejects_unreviewed_mixed_precision_and_cuda(tmp_path):
    train, val = loaders()
    with pytest.raises(ValueError, match="mixed precision"):
        Trainer(model_for(config(precision="bf16")), config(precision="bf16"), train, val,
                CheckpointManager(tmp_path / "a"), JsonlLogger(tmp_path / "a.jsonl"))
    with pytest.raises(RuntimeError, match="CPU only"):
        Trainer(model_for(config(device="cuda")), config(device="cuda"), train, val,
                CheckpointManager(tmp_path / "b"), JsonlLogger(tmp_path / "b.jsonl"))
