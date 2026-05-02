"""
Entry point for Ferrum Phase 1 training.

Usage:
    python train_pipeline.py --pgn path/to/lichess.pgn [--resume checkpoints/ckpt.pt]
"""
import argparse
import pathlib
import torch

import config
from board_encoder import encode
from data_loader import parse_pgn_to_hdf5, make_dataloader
from model import FerrumNet, load_checkpoint
from trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", nargs="+", required=True, help="PGN file paths")
    parser.add_argument("--hdf5", default="data/ferrum.h5", help="HDF5 output path")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    pgn_paths = [pathlib.Path(p) for p in args.pgn]
    hdf5_path = pathlib.Path(args.hdf5)

    print(f"[main] device={config.DEVICE} pgn_files={len(pgn_paths)} hdf5={hdf5_path}")

    if not hdf5_path.exists():
        print(f"Parsing PGN files -> {hdf5_path}")
        parse_pgn_to_hdf5(pgn_paths, hdf5_path)
    else:
        print(f"[main] reusing existing HDF5 dataset {hdf5_path}")

    train_loader = make_dataloader(hdf5_path, augment=True, split="train")
    val_loader   = make_dataloader(hdf5_path, augment=False, split="val")

    print(
        f"[main] loaders ready: train_samples={len(train_loader.dataset)} "
        f"val_samples={len(val_loader.dataset)}"
    )

    model = FerrumNet(config.FILTERS, config.NUM_BLOCKS).to(config.DEVICE)
    print(
        f"[main] model ready: filters={config.FILTERS} blocks={config.NUM_BLOCKS} "
        f"params={sum(p.numel() for p in model.parameters())}"
    )

    start_epoch, start_step, optimizer = 0, 0, None
    if args.resume:
        from trainer import build_optimizer
        optimizer = build_optimizer(model)
        meta = load_checkpoint(args.resume, model, optimizer)
        start_epoch = meta["epoch"]
        start_step  = meta["step"]
        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    train(model, train_loader, val_loader,
          start_epoch=start_epoch, start_step=start_step, optimizer=optimizer)


if __name__ == "__main__":
    main()
