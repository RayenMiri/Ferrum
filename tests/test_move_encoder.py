# tests/test_move_encoder.py
import chess
import torch
from move_encoder import move_to_index, index_to_move, legal_move_mask

def test_all_starting_moves_in_range():
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert 0 <= idx < 4672, f"Out-of-range {idx} for {move}"

def test_move_roundtrip_starting_position():
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        assert index_to_move(idx, board) == move, f"Roundtrip: {move} -> {idx}"

def test_knight_moves_use_knight_planes():
    board = chess.Board()
    for uci in ["g1f3", "g1h3", "b1a3", "b1c3"]:
        plane = move_to_index(chess.Move.from_uci(uci), board) // 64
        assert 56 <= plane < 64, f"{uci} got plane {plane}"

def test_queen_promotion_uses_queen_plane():
    board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
    assert move_to_index(chess.Move.from_uci("e7e8q"), board) // 64 < 56

def test_knight_underpromotion_uses_underpromotion_plane():
    board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
    plane = move_to_index(chess.Move.from_uci("e7e8n"), board) // 64
    assert 64 <= plane < 73

def test_distinct_moves_get_distinct_indices():
    board = chess.Board()
    idxs = [move_to_index(m, board) for m in board.legal_moves]
    assert len(idxs) == len(set(idxs))
