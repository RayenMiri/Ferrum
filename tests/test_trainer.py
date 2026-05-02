import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import pytest
from model import FerrumNet
from trainer import build_optimizer, build_scheduler, training_step, evaluate_accuracy
import config

config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_model():
    m = FerrumNet(filters=8, num_blocks=2)
    return m.to(config.DEVICE)

def make_batch(batch_size=4):
    state = torch.zeros(batch_size, 18, 8, 8, device=config.DEVICE)
    policy = torch.zeros(batch_size, dtype=torch.long, device=config.DEVICE)
    value = torch.zeros(batch_size, 1, device=config.DEVICE)
    mask = torch.zeros(batch_size, 4672, dtype=torch.bool, device=config.DEVICE)
    mask[:, :20] = True  # first 20 moves are legal, target is move 0
    return state, policy, value, mask

def test_build_optimizer():
    model = make_model()
    opt = build_optimizer(model)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == config.LR

def test_build_scheduler():
    model = make_model()
    opt = build_optimizer(model)
    sched = build_scheduler(opt, steps_per_epoch=100)
    opt.step()
    sched.step()

def test_build_scheduler_lr_values():
    model = make_model()
    opt = build_optimizer(model)
    steps_per_epoch = 10
    sched = build_scheduler(opt, steps_per_epoch=steps_per_epoch)
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

    # At step 0 (before first step call), lr should be minimal (warmup start)
    assert opt.param_groups[0]["lr"] < config.LR

    # After warmup_steps, lr should equal config.LR
    for _ in range(warmup_steps):
        opt.param_groups[0]["lr"] = config.LR  # avoid no-grad warning
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - config.LR) < 1e-6

def test_training_step_returns_loss():
    model = make_model()
    opt = build_optimizer(model)
    batch = make_batch()
    loss = training_step(model, batch, opt, config.DEVICE)
    assert isinstance(loss, float)
    assert loss > 0

def test_training_step_decreases_loss():
    """Overfit to a single batch — loss should decrease over 20 steps."""
    model = make_model()
    opt = build_optimizer(model)
    batch = make_batch()
    losses = [training_step(model, batch, opt, config.DEVICE) for _ in range(20)]
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

def test_evaluate_accuracy():
    model = make_model()
    from torch.utils.data import DataLoader, TensorDataset
    state, policy, value, mask = make_batch(batch_size=8)
    ds = TensorDataset(state, policy, value, mask)
    loader = DataLoader(ds, batch_size=4)
    top1, top5 = evaluate_accuracy(model, loader, config.DEVICE)
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top5 <= 1.0
    assert top5 >= top1

def test_train_writes_metrics(tmp_path, monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_cfg, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "MAX_EPOCHS", 1)
    monkeypatch.setattr(_cfg, "CHECKPOINT_STEPS", 10000)  # prevent checkpoint writes
    monkeypatch.setattr(_cfg, "CHECKPOINT_DIR", tmp_path)

    from torch.utils.data import DataLoader, TensorDataset
    from trainer import train

    state, policy, value, mask = make_batch(batch_size=4)
    ds = TensorDataset(state, policy, value, mask)
    loader = DataLoader(ds, batch_size=4)

    model = make_model()
    train(model, loader, loader)

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    import json
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) >= {"epoch", "step", "train_loss", "val_top1", "val_top5"}
