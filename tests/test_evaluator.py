import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from model import FerrumNet
from evaluator import compute_perplexity
import config

config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model():
    m = FerrumNet(filters=8, num_blocks=2)
    return m.to(config.DEVICE)


def make_loader(batch_size=8):
    state = torch.zeros(batch_size, 18, 8, 8, device=config.DEVICE)
    policy = torch.zeros(batch_size, dtype=torch.long, device=config.DEVICE)
    value = torch.zeros(batch_size, 1, device=config.DEVICE)
    mask = torch.ones(batch_size, 4672, dtype=torch.bool, device=config.DEVICE)
    ds = TensorDataset(state, policy, value, mask)
    return DataLoader(ds, batch_size=4)


def test_perplexity_is_positive():
    model = make_model()
    loader = make_loader()
    ppl = compute_perplexity(model, loader, config.DEVICE)
    assert isinstance(ppl, float)
    assert ppl > 1.0


def test_perplexity_decreases_after_overfit():
    """Overfit a tiny model on one batch; perplexity should drop."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    model = make_model()
    loader = make_loader(batch_size=4)
    ppl_before = compute_perplexity(model, loader, config.DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    state = torch.zeros(16, 18, 8, 8, device=config.DEVICE)
    policy_target = torch.zeros(16, dtype=torch.long, device=config.DEVICE)
    mask = torch.ones(16, 4672, dtype=torch.bool, device=config.DEVICE)
    for _ in range(50):
        logits, _ = model(state)
        logits = logits.masked_fill(~mask, -torch.inf)
        loss = F.cross_entropy(logits, policy_target)
        opt.zero_grad(); loss.backward(); opt.step()

    ppl_after = compute_perplexity(model, loader, config.DEVICE)
    assert ppl_after < ppl_before, f"Perplexity did not decrease: {ppl_before:.2f} -> {ppl_after:.2f}"


def test_probe_elo_importable():
    """probe_elo_vs_stockfish must be importable and have the right signature."""
    from evaluator import probe_elo_vs_stockfish
    import inspect
    sig = inspect.signature(probe_elo_vs_stockfish)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "stockfish_path" in params
    assert "num_games" in params
