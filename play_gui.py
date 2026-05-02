"""
Graphical chess play against a Ferrum checkpoint or Stockfish.

Usage examples:
    python play_gui.py --mode checkpoint --checkpoint checkpoints/ckpt_epoch0_step5000.pt
    python play_gui.py --mode stockfish --stockfish D:\\stockfish\\stockfish-windows-x86-64-avx2.exe
    python play_gui.py --mode auto --checkpoint checkpoints/ckpt_epoch0_step5000.pt --stockfish D:\\stockfish\\stockfish-windows-x86-64-avx2.exe
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch

import config
from board_encoder import encode
from model import FerrumNet, load_checkpoint
from move_encoder import index_to_move, legal_move_mask

ASSET_NAMES = {
    (chess.PAWN, chess.WHITE): "wP.png",
    (chess.KNIGHT, chess.WHITE): "wN.png",
    (chess.BISHOP, chess.WHITE): "wB.png",
    (chess.ROOK, chess.WHITE): "wR.png",
    (chess.QUEEN, chess.WHITE): "wQ.png",
    (chess.KING, chess.WHITE): "wK.png",
    (chess.PAWN, chess.BLACK): "bP.png",
    (chess.KNIGHT, chess.BLACK): "bN.png",
    (chess.BISHOP, chess.BLACK): "bB.png",
    (chess.ROOK, chess.BLACK): "bR.png",
    (chess.QUEEN, chess.BLACK): "bQ.png",
    (chess.KING, chess.BLACK): "bK.png",
}


@dataclass
class OpponentConfig:
    mode: str
    checkpoint: Path | None = None
    stockfish: Path | None = None
    stockfish_depth: int = 1


class FerrumOpponent:
    def __init__(self, cfg: OpponentConfig):
        self.cfg = cfg
        self.model: FerrumNet | None = None
        self.engine: chess.engine.SimpleEngine | None = None
        self.label = "Ferrum" if cfg.mode == "checkpoint" else "Stockfish"

        if cfg.mode == "checkpoint":
            if cfg.checkpoint is None:
                raise ValueError("checkpoint mode requires --checkpoint")
            self.model = FerrumNet(config.FILTERS, config.NUM_BLOCKS).to(config.DEVICE)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LR)
            load_checkpoint(cfg.checkpoint, self.model, optimizer)
            self.model.eval()
        elif cfg.mode == "stockfish":
            if cfg.stockfish is None:
                raise ValueError("stockfish mode requires --stockfish")
            self.engine = chess.engine.SimpleEngine.popen_uci(str(cfg.stockfish))
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass

    def choose_move(self, board: chess.Board) -> chess.Move:
        if self.cfg.mode == "checkpoint":
            assert self.model is not None
            return self._choose_checkpoint_move(board)
        assert self.engine is not None
        return self._choose_stockfish_move(board)

    def _choose_checkpoint_move(self, board: chess.Board) -> chess.Move:
        state = encode(board).unsqueeze(0).to(config.DEVICE)
        mask = legal_move_mask(board).to(config.DEVICE)
        with torch.no_grad():
            logits, _ = self.model(state)
            logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
            idx = int(torch.argmax(logits).item())

        move = index_to_move(idx, board)
        if move not in board.legal_moves:
            move = list(board.legal_moves)[0]
        return move

    def _choose_stockfish_move(self, board: chess.Board) -> chess.Move:
        assert self.engine is not None
        result = self.engine.play(board, chess.engine.Limit(depth=self.cfg.stockfish_depth))
        if result.move is None:
            return list(board.legal_moves)[0]
        return result.move


class ChessGUI:
    def __init__(
        self,
        root: tk.Tk,
        black_player: FerrumOpponent | None,
        white_player: FerrumOpponent | None,
        human_color: str | None,
    ):
        self.root = root
        self.white_player = white_player
        self.black_player = black_player
        self.auto_mode = human_color is None
        self.human_is_white = None if human_color is None else human_color == "white"
        self.board = chess.Board()
        self.selected_square: int | None = None
        self.legal_targets: set[int] = set()
        self.thinking = False
        self.game_over = False
        self.asset_dir = Path(__file__).resolve().parent / "assets"
        self.piece_images = self._load_piece_images()

        if self.auto_mode:
            self.root.title(f"{self._player_label(chess.WHITE)} vs {self._player_label(chess.BLACK)}")
        else:
            self.root.title(f"Ferrum Chess - You vs {self._player_label(not self.human_is_white)}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status = tk.StringVar()
        self.status.set(self._status_text())

        top = tk.Frame(root)
        top.pack(padx=10, pady=10)

        self.buttons: list[list[tk.Button]] = []
        board_frame = tk.Frame(top)
        board_frame.config(width=84 * 8, height=84 * 8)
        board_frame.grid(row=0, column=0, rowspan=2)
        board_frame.grid_propagate(False)

        square_size = 84
        for index in range(8):
            board_frame.grid_columnconfigure(index, minsize=square_size, weight=1)
            board_frame.grid_rowconfigure(index, minsize=square_size, weight=1)

        for rank in range(7, -1, -1):
            row: list[tk.Button] = []
            for file in range(8):
                square = chess.square(file, rank)
                btn = tk.Button(
                    board_frame,
                    command=lambda sq=square: self.on_square_click(sq),
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                    padx=0,
                    pady=0,
                )
                btn.grid(row=7 - rank, column=file, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)

        side = tk.Frame(top)
        side.grid(row=0, column=1, sticky="n", padx=(12, 0))

        tk.Label(side, textvariable=self.status, justify="left", wraplength=300).pack(anchor="w")

        controls = tk.Frame(side)
        controls.pack(anchor="w", pady=(10, 10))

        tk.Button(controls, text="New Game", command=self.new_game).grid(row=0, column=0, sticky="ew")
        if not self.auto_mode:
            tk.Button(controls, text="Flip Human Side", command=self.flip_side).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.move_log = tk.Text(side, width=32, height=18, state="disabled")
        self.move_log.pack(anchor="w")

        self.refresh_board()
        self.root.after(250, self.request_opponent_move)

    def on_close(self) -> None:
        if self.white_player is not None:
            self.white_player.close()
        if self.black_player is not None:
            self.black_player.close()
        self.root.destroy()

    def flip_side(self) -> None:
        if self.auto_mode or self.thinking or self.game_over:
            return
        self.human_is_white = not self.human_is_white
        self.root.title(f"Ferrum Chess - You vs {self._player_label(not self.human_is_white)}")
        self.new_game()

    def new_game(self) -> None:
        if self.thinking:
            return
        self.board.reset()
        self.selected_square = None
        self.legal_targets = set()
        self.game_over = False
        self.clear_log()
        self.refresh_board()
        self.root.after(250, self.request_opponent_move)

    def on_square_click(self, square: int) -> None:
        if self.auto_mode or self.game_over or self.thinking:
            return
        if not self._is_human_turn():
            return

        piece = self.board.piece_at(square)

        if self.selected_square is None:
            if piece is not None and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_targets = self._legal_targets_for(square)
                self.refresh_board()
            return

        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = set()
            self.refresh_board()
            return

        move = self._build_move(self.selected_square, square)
        if move is not None and move in self.board.legal_moves:
            self.board.push(move)
            self.append_log(f"You: {move}")
            self.selected_square = None
            self.legal_targets = set()
            self.refresh_board()
            self._check_game_over()
            if not self.game_over:
                self.root.after(150, self.request_opponent_move)
            return

        if piece is not None and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_targets = self._legal_targets_for(square)
            self.refresh_board()
        else:
            self.selected_square = None
            self.legal_targets = set()
            self.refresh_board()

    def request_opponent_move(self) -> None:
        if self.game_over or self.thinking:
            return
        player = self._player_for_turn()
        if player is None:
            return

        self.thinking = True
        self.status.set(self._status_text(prefix=f"{player.label} thinking..."))
        board_copy = self.board.copy(stack=False)

        def worker() -> None:
            try:
                move = player.choose_move(board_copy)
                error = None
            except Exception as exc:
                move = None
                error = exc

            def finish() -> None:
                self.thinking = False
                if error is not None:
                    self.status.set(f"Engine error: {error}")
                    return
                if move is not None and move in self.board.legal_moves:
                    self.board.push(move)
                    self.append_log(f"{player.label}: {move}")
                else:
                    self.append_log(f"{player.label}: no legal move")
                self.refresh_board()
                self._check_game_over()

                if not self.game_over:
                    self.root.after(150, self.request_opponent_move)

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _legal_targets_for(self, square: int) -> set[int]:
        targets = set()
        for move in self.board.legal_moves:
            if move.from_square == square:
                targets.add(move.to_square)
        return targets

    def _is_human_turn(self) -> bool:
        return self.human_is_white is not None and ((self.board.turn == chess.WHITE) == self.human_is_white)

    def _player_for_turn(self) -> FerrumOpponent | None:
        return self.white_player if self.board.turn == chess.WHITE else self.black_player

    def _player_label(self, color: bool) -> str:
        player = self.white_player if color == chess.WHITE else self.black_player
        if self.auto_mode:
            return player.label if player is not None else "Human"
        if self.human_is_white is not None and ((color == chess.WHITE) == self.human_is_white):
            return "You"
        return player.label if player is not None else "Human"

    def _load_piece_images(self) -> dict[tuple[int, bool], tk.PhotoImage]:
        images: dict[tuple[int, bool], tk.PhotoImage] = {}
        for key, file_name in ASSET_NAMES.items():
            path = self.asset_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing GUI asset: {path}")
            images[key] = tk.PhotoImage(master=self.root, file=str(path))
        return images

    def _build_move(self, from_square: int, to_square: int) -> chess.Move | None:
        piece = self.board.piece_at(from_square)
        if piece is None:
            return None

        promotion = None
        if piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if to_rank in {0, 7}:
                promotion = chess.QUEEN

        move = chess.Move(from_square, to_square, promotion=promotion)
        return move if move in self.board.legal_moves else None

    def _check_game_over(self) -> None:
        outcome = self.board.outcome()
        if outcome is None:
            self.status.set(self._status_text())
            return

        self.game_over = True
        if outcome.winner is None:
            result = "Game over: draw"
        elif outcome.winner == (chess.WHITE if self.human_is_white else chess.BLACK):
            result = "Game over: you won"
        else:
            result = "Game over: Ferrum won"
        self.status.set(result)

    def _status_text(self, prefix: str | None = None) -> str:
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        if self.auto_mode:
            lines = [
                f"White: {self._player_label(chess.WHITE)}",
                f"Black: {self._player_label(chess.BLACK)}",
                f"Turn: {self._player_label(self.board.turn)}",
            ]
        else:
            side = "White" if self.human_is_white else "Black"
            lines = [f"You are playing: {side}", f"Opponent: {self._player_label(not self.human_is_white)}", f"Turn: {turn}"]
        if prefix:
            lines.insert(0, prefix)
        return "\n".join(lines)

    def refresh_board(self) -> None:
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)
                btn = self.buttons[rank][file]
                piece = self.board.piece_at(square)
                if piece is None:
                    btn.config(text="", image="")
                else:
                    btn.config(text="", image=self.piece_images[(piece.piece_type, piece.color)])

                is_light = (file + (7 - rank)) % 2 == 0
                base_bg = "#f0d9b5" if is_light else "#b58863"
                if square == self.selected_square:
                    bg = "#f7f769"
                elif square in self.legal_targets:
                    bg = "#89d68b"
                else:
                    bg = base_bg
                btn.config(bg=bg, activebackground=bg)

        self.status.set(self._status_text() if not self.game_over else self.status.get())

    def append_log(self, text: str) -> None:
        self.move_log.config(state="normal")
        self.move_log.insert("end", text + "\n")
        self.move_log.see("end")
        self.move_log.config(state="disabled")

    def clear_log(self) -> None:
        self.move_log.config(state="normal")
        self.move_log.delete("1.0", "end")
        self.move_log.config(state="disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graphical play against a Ferrum checkpoint or Stockfish.")
    parser.add_argument("--mode", choices=("checkpoint", "stockfish", "auto"), required=True)
    parser.add_argument("--checkpoint", help="Checkpoint path for checkpoint mode")
    parser.add_argument("--stockfish", help="Path to Stockfish executable for stockfish mode")
    parser.add_argument("--color", choices=("white", "black"), default="white", help="Human side")
    parser.add_argument("--ferrum-color", choices=("white", "black"), default="white", help="Ferrum side in auto mode")
    parser.add_argument("--depth", type=int, default=1, help="Stockfish search depth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    white_player: FerrumOpponent | None = None
    black_player: FerrumOpponent | None = None
    human_color: str | None = None

    try:
        if args.mode == "auto":
            if args.checkpoint is None or args.stockfish is None:
                raise ValueError("auto mode requires both --checkpoint and --stockfish")

            ferrum_player = FerrumOpponent(
                OpponentConfig(
                    mode="checkpoint",
                    checkpoint=Path(args.checkpoint),
                )
            )
            stockfish_player = FerrumOpponent(
                OpponentConfig(
                    mode="stockfish",
                    stockfish=Path(args.stockfish),
                    stockfish_depth=args.depth,
                )
            )

            if args.ferrum_color == "white":
                white_player = ferrum_player
                black_player = stockfish_player
            else:
                white_player = stockfish_player
                black_player = ferrum_player
        else:
            opponent = FerrumOpponent(
                OpponentConfig(
                    mode=args.mode,
                    checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                    stockfish=Path(args.stockfish) if args.stockfish else None,
                    stockfish_depth=args.depth,
                )
            )
            if args.color == "white":
                black_player = opponent
                human_color = "white"
            else:
                white_player = opponent
                human_color = "black"

        root = tk.Tk()
        if args.mode == "auto":
            ChessGUI(root, white_player, black_player, human_color=None)
        else:
            ChessGUI(root, white_player, black_player, human_color=human_color)

        root.mainloop()
    finally:
        if white_player is not None:
            white_player.close()
        if black_player is not None:
            black_player.close()


if __name__ == "__main__":
    main()