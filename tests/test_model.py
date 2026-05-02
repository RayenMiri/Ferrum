import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch
import pytest
from model import ResBlock, FerrumNet, save_checkpoint, load_checkpoint
import config, tempfile

def make_model():
    return FerrumNet(filters=config.FILTERS, num_blocks=config.NUM_BLOCKS)

def test_resblock_output_shape():
    block = ResBlock(filters=32)
    x = torch.zeros(2, 32, 8, 8)
    out = block(x)
    assert out.shape == (2, 32, 8, 8)

def test_ferrumnet_policy_shape():
    model = make_model()
    x = torch.zeros(4, 18, 8, 8)
    policy, value = model(x)
    assert policy.shape == (4, 4672)

def test_ferrumnet_value_shape():
    model = make_model()
    x = torch.zeros(4, 18, 8, 8)
    policy, value = model(x)
    assert value.shape == (4, 1)

def test_value_in_tanh_range():
    model = make_model()
    x = torch.randn(8, 18, 8, 8)
    _, value = model(x)
    assert value.min() >= -1.0 - 1e-6
    assert value.max() <=  1.0 + 1e-6

def test_checkpoint_roundtrip(tmp_path):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, epoch=1, step=100, path=ckpt_path)
    assert ckpt_path.exists()
    meta = load_checkpoint(ckpt_path, model, opt)
    assert meta["epoch"] == 1
    assert meta["step"] == 100

def test_policy_spatial_reshape():
    model = make_model()
    x = torch.zeros(4, 18, 8, 8)
    policy, _ = model(x)
    reshaped = policy.view(4, 73, 8, 8)
    assert reshaped.shape == (4, 73, 8, 8)

def test_checkpoint_architecture_mismatch(tmp_path):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    # Save with wrong config_snapshot
    import torch as _t
    _t.save({
        "epoch": 1, "step": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config_snapshot": {"FILTERS": 999, "NUM_BLOCKS": 999},
    }, ckpt_path)
    with pytest.raises(ValueError):
        load_checkpoint(ckpt_path, model, opt)
