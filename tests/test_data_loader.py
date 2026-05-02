# tests/test_data_loader.py
import io
from pathlib import Path
import chess.pgn, h5py, pytest, torch

SAMPLE_PGN = """\
[Event "Rated Classical game"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "1800+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "Rated Blitz game"]
[WhiteElo "1900"]
[BlackElo "1850"]
[TimeControl "300+3"]
[Result "0-1"]

1. d4 d5 0-1

[Event "Rated Classical game"]
[WhiteElo "900"]
[BlackElo "800"]
[TimeControl "1800+0"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2
"""

UNRATED_PGN = """\
[Event "Casual Classical game"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "1800+0"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

ABANDONED_PGN = """\
[Event "Rated Classical game"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "1800+0"]
[Result "*"]

1. e4 e5 2. Nf3 *
"""


def _games(text):
    buf = io.StringIO(text)
    gs = []
    while (g := chess.pgn.read_game(buf)) is not None:
        gs.append(g)
    return gs


def _make_h5(tmp_path):
    from data_loader import parse_pgn_to_hdf5
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(SAMPLE_PGN)
    h5_path = tmp_path / "out.h5"
    parse_pgn_to_hdf5([pgn_path], h5_path)
    return h5_path


def test_filter_accepts_classical_in_range():
    from data_loader import _game_passes_filter
    assert _game_passes_filter(_games(SAMPLE_PGN)[0]) is True


def test_filter_rejects_blitz():
    from data_loader import _game_passes_filter
    assert _game_passes_filter(_games(SAMPLE_PGN)[1]) is False


def test_filter_rejects_low_elo():
    from data_loader import _game_passes_filter
    assert _game_passes_filter(_games(SAMPLE_PGN)[2]) is False


def test_filter_rejects_unrated():
    from data_loader import _game_passes_filter
    game = _games(UNRATED_PGN)[0]
    assert _game_passes_filter(game) is False


def test_filter_rejects_abandoned():
    from data_loader import _game_passes_filter
    game = _games(ABANDONED_PGN)[0]
    assert _game_passes_filter(game) is False


def test_hdf5_shapes(tmp_path):
    h5_path = _make_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        # With only 1 qualifying game, all positions go to train
        grp = h5["train"]
        n = grp["positions"].shape[0]
        assert n > 0
        assert grp["positions"].shape == (n, 18, 8, 8)
        assert grp["policy_targets"].shape == (n,)
        assert grp["value_targets"].shape == (n,)
        assert grp["legal_masks"].shape == (n, 4672)


def test_hdf5_value_ranges(tmp_path):
    h5_path = _make_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        grp = h5["train"]
        assert ((grp["value_targets"][:] >= -1.0) & (grp["value_targets"][:] <= 1.0)).all()
        assert ((grp["policy_targets"][:] >= 0) & (grp["policy_targets"][:] < 4672)).all()


def test_ferrum_dataset_getitem(tmp_path):
    from data_loader import FerrumDataset
    h5_path = _make_h5(tmp_path)
    ds = FerrumDataset(h5_path, augment=False, split="train")
    assert len(ds) > 0
    state, policy, value, mask = ds[0]
    assert state.shape == (18, 8, 8)
    assert 0 <= policy < 4672
    assert value in (-1.0, 0.0, 1.0)
    assert mask.shape == (4672,) and mask.dtype == torch.bool


def test_augment_preserves_policy_and_mask(tmp_path):
    from data_loader import FerrumDataset
    h5 = _make_h5(tmp_path)
    ds = FerrumDataset(h5, split="train", augment=True)
    state, policy, value, mask = ds[0]
    assert 0 <= policy.item() < 4672
    # mask bit count unchanged after flip
    ds_no_aug = FerrumDataset(h5, split="train", augment=False)
    _, _, _, mask_no_aug = ds_no_aug[0]
    assert mask.sum() == mask_no_aug.sum()


def test_augment_train_only(tmp_path):
    from data_loader import FerrumDataset
    h5 = _make_h5(tmp_path)
    with pytest.raises(ValueError):
        FerrumDataset(h5, split="val", augment=True)
