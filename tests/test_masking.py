# tests/test_masking.py
import chess
import torch
from move_encoder import legal_move_mask, move_to_index

def test_mask_shape_and_dtype():
    mask = legal_move_mask(chess.Board())
    assert mask.shape == (4672,) and mask.dtype == torch.bool

def test_mask_count_starting_position():
    assert legal_move_mask(chess.Board()).sum().item() == 20

def test_mask_count_matches_legal_moves():
    for fen in [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/4P3/8/8/8/8/8/4K3 w - - 0 1",
    ]:
        board = chess.Board(fen)
        n = len(list(board.legal_moves))
        assert legal_move_mask(board).sum().item() == n, fen

def test_all_legal_moves_set_in_mask():
    board = chess.Board()
    mask = legal_move_mask(board)
    for move in board.legal_moves:
        assert mask[move_to_index(move, board)], f"{move} missing from mask"
