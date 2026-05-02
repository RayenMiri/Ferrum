"""
Play a terminal chess game against a saved Ferrum checkpoint.

Usage:
    python play_against_checkpoint.py --checkpoint checkpoints/ckpt_epoch0_step1000.pt
"""
import argparse
from pathlib import Path

import chess
import torch

import config
from board_encoder import encode
from model import FerrumNet, load_checkpoint
from move_encoder import legal_move_mask, index_to_move


def _pick_model_move(model: FerrumNet, board: chess.Board, device: str) -> chess.Move:
    state = encode(board).unsqueeze(0).to(device)
    mask = legal_move_mask(board).to(device)

    with torch.no_grad():
        logits, _ = model(state)
        logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
        idx = int(torch.argmax(logits).item())

    move = index_to_move(idx, board)
    if move not in board.legal_moves:
        move = list(board.legal_moves)[0]
    return move


def _read_human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move (uci or san, 'quit' to stop): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit(0)

        try:
            if len(raw) in {4, 5} and raw[0].isalpha() and raw[2].isalpha():
                move = chess.Move.from_uci(raw)
                if move in board.legal_moves:
                    return move
            move = board.parse_san(raw)
            if move in board.legal_moves:
                return move
        except Exception:
            pass

        print("Invalid move. Try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against a Ferrum checkpoint in the terminal.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Ferrum checkpoint file")
    parser.add_argument(
        "--color",
        choices=("white", "black"),
        default="white",
        help="Which side the human plays",
    )
    parser.add_argument("--device", default=config.DEVICE, help="torch device to use")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = FerrumNet(config.FILTERS, config.NUM_BLOCKS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    meta = load_checkpoint(checkpoint_path, model, optimizer)
    model.eval()

    human_is_white = args.color == "white"
    board = chess.Board()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Starting at epoch={meta['epoch']} step={meta['step']}")
    print(f"You are playing as {'White' if human_is_white else 'Black'}")

    while not board.is_game_over():
        print()
        print(board)
        print()

        human_turn = (board.turn == chess.WHITE) == human_is_white
        if human_turn:
            move = _read_human_move(board)
            board.push(move)
            print(f"You played: {move}")
        else:
            move = _pick_model_move(model, board, args.device)
            board.push(move)
            print(f"Ferrum played: {move}")

    print()
    print(board)
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        print("Game over: draw")
    elif outcome.winner == (chess.WHITE if human_is_white else chess.BLACK):
        print("Game over: you won")
    else:
        print("Game over: Ferrum won")


if __name__ == "__main__":
    main()