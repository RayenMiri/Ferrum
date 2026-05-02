# data_loader.py
import chess
import chess.pgn
import h5py
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


class _SplitWriter:
    def __init__(self, h5_file: h5py.File, group_name: str, chunk_size: int):
        self._grp = h5_file.require_group(group_name)
        self._chunk_size = chunk_size
        self._pos_buf, self._pol_buf = [], []
        self._val_buf, self._msk_buf = [], []

    def add(self, state, policy, value, mask):
        self._pos_buf.append(state)
        self._pol_buf.append(policy)
        self._val_buf.append(value)
        self._msk_buf.append(mask)
        if len(self._pos_buf) >= self._chunk_size:
            self.flush()

    def flush(self):
        if not self._pos_buf:
            return
        data = {
            "positions":      np.stack(self._pos_buf).astype(np.float32),
            "policy_targets": np.array(self._pol_buf, dtype=np.int32),
            "value_targets":  np.array(self._val_buf, dtype=np.float32),
            "legal_masks":    np.stack(self._msk_buf).astype(bool),
        }
        n = len(self._pos_buf)
        for name, arr in data.items():
            if name not in self._grp:
                self._grp.create_dataset(name, data=arr, maxshape=(None,) + arr.shape[1:], chunks=True)
            else:
                ds = self._grp[name]
                old = ds.shape[0]
                ds.resize(old + n, axis=0)
                ds[old:] = arr
        self._pos_buf.clear()
        self._pol_buf.clear()
        self._val_buf.clear()
        self._msk_buf.clear()


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
    """Stream PGN files into train/val/test HDF5 splits with low memory use."""
    print(f"[data] streaming {len(pgn_paths)} PGN file(s) -> {out_path}")
    train_prob, val_prob, _ = split
    progress_interval = 100_000
    processed = accepted = filtered = malformed = 0

    with h5py.File(out_path, "w") as h5:
        writers = {
            "train": _SplitWriter(h5, "train", chunk_size),
            "val": _SplitWriter(h5, "val", chunk_size),
            "test": _SplitWriter(h5, "test", chunk_size),
        }

        for pgn_path in pgn_paths:
            print(f"[data] reading {pgn_path}")
            with open(pgn_path, errors="replace") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    processed += 1
                    if processed % progress_interval == 0:
                        print(
                            f"[data] processed={processed} accepted={accepted} "
                            f"filtered={filtered} malformed={malformed}"
                        )

                    if not _game_passes_filter(game):
                        filtered += 1
                        continue

                    try:
                        positions = list(_extract_positions(game))
                    except Exception as exc:
                        malformed += 1
                        print(f"[warn] skipping malformed game #{processed} in {pgn_path}: {exc}")
                        continue

                    if not positions:
                        continue

                    if accepted == 0:
                        writer = writers["train"]
                    elif accepted == 1 and val_prob > 0:
                        writer = writers["val"]
                    elif accepted == 2 and split[2] > 0:
                        writer = writers["test"]
                    else:
                        split_roll = random.random()
                        if split_roll < train_prob:
                            writer = writers["train"]
                        elif split_roll < train_prob + val_prob:
                            writer = writers["val"]
                        else:
                            writer = writers["test"]

                    for t, p, v, m in positions:
                        writer.add(t, p, v, m)
                    accepted += 1

        for writer in writers.values():
            writer.flush()

    print(
        f"[data] done processed={processed} accepted={accepted} "
        f"filtered={filtered} malformed={malformed}"
    )
    print(f"[data] wrote HDF5 dataset to {out_path}")


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
        print(
            f"[data] loaded split='{split}' augment={augment} "
            f"samples={self._len} from {h5_path}"
        )

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
    print(
        f"[data] dataloader split='{split}' batch_size={batch_size or config.BATCH_SIZE} "
        f"shuffle={augment} workers={config.NUM_WORKERS} pin_memory={torch.cuda.is_available()}"
    )
    return DataLoader(ds, batch_size=batch_size or config.BATCH_SIZE,
                      shuffle=augment, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
