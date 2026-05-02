# data_loader.py
import chess
import chess.pgn
import h5py
import multiprocessing
import numpy as np
import os
import random
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

import config
from board_encoder import encode, _flip_move_index
from move_encoder import move_to_index, legal_move_mask

_CLASSICAL_RAPID = {"classical", "rapid"}


def _time_control_type(tc: str) -> str:
    if not tc or tc == "-":
        return "unknown"
    try:
        base = int(tc.split("+")[0].split("/")[-1])
    except ValueError:
        return "unknown"
    if base >= 900: return "classical"
    if base >= 480: return "rapid"
    if base >= 180: return "blitz"
    return "bullet"


def _game_passes_filter(game: chess.pgn.Game) -> bool:
    h = game.headers
    event = h.get("Event", "")
    if "rated" not in event.lower():
        return False
    result = game.headers.get("Result", "*")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return False
    try:
        w, b = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
    except ValueError:
        return False
    if not (config.ELO_MIN <= w <= config.ELO_MAX): return False
    if not (config.ELO_MIN <= b <= config.ELO_MAX): return False
    return _time_control_type(h.get("TimeControl", "")) in _CLASSICAL_RAPID


def _outcome_value(result: str, color: chess.Color) -> float:
    if result == "1-0": return 1.0 if color == chess.WHITE else -1.0
    if result == "0-1": return -1.0 if color == chess.WHITE else 1.0
    return 0.0


def _extract_positions(game: chess.pgn.Game):
    result = game.headers.get("Result", "*")
    board  = game.board()
    for move in game.mainline_moves():
        if not board.is_legal(move):
            break
        yield (encode(board).numpy(), move_to_index(move, board),
               _outcome_value(result, board.turn), legal_move_mask(board).numpy())
        board.push(move)


def _process_pgn_file(pgn_path: Path):
    """Worker function: parse one PGN file and return list of per-game position tuples."""
    games_positions = []
    with open(pgn_path, errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if not _game_passes_filter(game):
                continue
            positions = list(_extract_positions(game))
            if positions:
                games_positions.append(positions)
    return games_positions


def _write_split_to_hdf5(h5_file: h5py.File, group_name: str,
                          games: list, chunk_size: int = 4096):
    """Write positions from a list of games into an HDF5 group."""
    grp = h5_file.require_group(group_name)
    pos_buf, pol_buf, val_buf, msk_buf = [], [], [], []

    def _flush():
        if not pos_buf:
            return
        data = {
            "positions":      np.stack(pos_buf).astype(np.float32),
            "policy_targets": np.array(pol_buf, dtype=np.int32),
            "value_targets":  np.array(val_buf, dtype=np.float32),
            "legal_masks":    np.stack(msk_buf).astype(bool),
        }
        n = len(pos_buf)
        for name, arr in data.items():
            if name not in grp:
                grp.create_dataset(name, data=arr, maxshape=(None,) + arr.shape[1:], chunks=True)
            else:
                ds = grp[name]; old = ds.shape[0]
                ds.resize(old + n, axis=0); ds[old:] = arr
        pos_buf.clear(); pol_buf.clear(); val_buf.clear(); msk_buf.clear()

    for game_positions in games:
        for t, p, v, m in game_positions:
            pos_buf.append(t); pol_buf.append(p)
            val_buf.append(v); msk_buf.append(m)
            if len(pos_buf) >= chunk_size:
                _flush()
    _flush()


def parse_pgn_to_hdf5(pgn_paths: List[Path], out_path: Path,
                       chunk_size: int = 4096,
                       split: Tuple[float, float, float] = (0.9, 0.05, 0.05)):
    """Parse PGN files in parallel and write train/val/test splits to one HDF5 file."""
    # Process each PGN file in parallel
    with multiprocessing.Pool(processes=config.NUM_WORKERS) as pool:
        results = pool.map(_process_pgn_file, pgn_paths)
    for i, r in enumerate(results):
        if not r:
            print(f"[warn] no positions from {pgn_paths[i]} — file empty, filtered out, or worker error")

    # Flatten: results is list-of-lists-of-games; combine all games
    all_games = []
    for file_games in results:
        all_games.extend(file_games)

    # Split by game, not by position
    n = len(all_games)
    if n == 0:
        train_games, val_games, test_games = [], [], []
    elif n == 1:
        train_games, val_games, test_games = all_games, [], []
    elif n == 2:
        train_games, val_games, test_games = all_games[:1], all_games[1:], []
    else:
        train_end = max(1, round(n * split[0]))
        val_end   = train_end + max(0, round(n * split[1]))
        val_end   = min(val_end, n - 1)  # leave at least one for test if n >= 3
        train_games = all_games[:train_end]
        val_games   = all_games[train_end:val_end]
        test_games  = all_games[val_end:]

    with h5py.File(out_path, "w") as h5:
        _write_split_to_hdf5(h5, "train", train_games, chunk_size)
        _write_split_to_hdf5(h5, "val",   val_games,   chunk_size)
        _write_split_to_hdf5(h5, "test",  test_games,  chunk_size)


class FerrumDataset(Dataset):
    def __init__(self, h5_path: Path, augment: bool = False, split: str = "train"):
        if augment and split != "train":
            raise ValueError("augment=True is only valid for the train split")
        self._h5_path = h5_path
        self._augment = augment
        self._split = split
        with h5py.File(h5_path, "r") as h5:
            grp = h5[split]
            self._len = grp["positions"].shape[0] if "positions" in grp else 0
        self._h5 = None

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def __len__(self): return self._len

    def __getitem__(self, idx):
        self._open()
        grp    = self._h5[self._split]
        state  = torch.from_numpy(grp["positions"][idx])
        policy = torch.tensor(int(grp["policy_targets"][idx]), dtype=torch.long)
        value  = torch.tensor(float(grp["value_targets"][idx]), dtype=torch.float32).unsqueeze(0)
        mask   = torch.from_numpy(grp["legal_masks"][idx])
        if self._augment and random.random() < 0.5:
            state  = state.flip(2)
            policy = torch.tensor(_flip_move_index(int(policy)), dtype=torch.long)
            new_mask = torch.zeros(4672, dtype=torch.bool)
            for i in mask.nonzero(as_tuple=False).squeeze(1).tolist():
                new_mask[_flip_move_index(i)] = True
            mask = new_mask
        return state, policy, value, mask


def make_dataloader(h5_path: Path, augment: bool, batch_size: int = None,
                    split: str = "train") -> DataLoader:
    ds = FerrumDataset(h5_path, augment=augment, split=split)
    return DataLoader(ds, batch_size=batch_size or config.BATCH_SIZE,
                      shuffle=augment, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
