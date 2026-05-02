# tests/test_encoder.py
import chess
import torch
from board_encoder import encode

def test_output_shape_and_dtype():
    t = encode(chess.Board())
    assert t.shape == (18, 8, 8)
    assert t.dtype == torch.float32

def test_white_pawns_on_rank_2():
    t = encode(chess.Board())
    assert t[0, 1, :].sum() == 8.0  # plane 0 = white pawns, row 1 = rank 2
    assert t[0, 0, :].sum() == 0.0

def test_white_king_on_e1():
    t = encode(chess.Board())
    assert t[5, 0, 4] == 1.0  # plane 5 = white king; e1 = row 0, col 4

def test_white_queen_on_d1():
    t = encode(chess.Board())
    assert t[4, 0, 3] == 1.0  # plane 4 = white queen; d1 = row 0, col 3

def test_black_pawns_on_rank_7():
    t = encode(chess.Board())
    assert t[6, 6, :].sum() == 8.0  # plane 6 = black pawns; rank 7 = row 6

def test_black_king_on_e8():
    t = encode(chess.Board())
    assert t[11, 7, 4] == 1.0  # plane 11 = black king; e8 = row 7, col 4

def test_piece_planes_empty_board():
    t = encode(chess.Board(fen=None))
    assert t[:12].sum() == 0.0

def test_castling_plane_starting():
    t = encode(chess.Board())
    assert t[12, 0, 7] == 1.0 and t[12, 0, 0] == 1.0
    assert t[12, 7, 7] == 1.0 and t[12, 7, 0] == 1.0

def test_castling_plane_no_rights():
    t = encode(chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1"))
    assert t[12].sum() == 0.0

def test_en_passant_plane_present():
    board = chess.Board()
    board.push_san("e4")  # ep target = e3 = row 2, col 4
    t = encode(board)
    assert t[13, 2, 4] == 1.0 and t[13].sum() == 1.0

def test_en_passant_plane_absent():
    assert encode(chess.Board())[13].sum() == 0.0

def test_fifty_move_plane():
    board = chess.Board()
    board.halfmove_clock = 50
    t = encode(board)
    assert abs(t[14, 0, 0].item() - 0.5) < 1e-6
    assert (t[14] == t[14, 0, 0]).all()

def test_side_to_move_white():
    assert (encode(chess.Board())[16] == 1.0).all()

def test_side_to_move_black():
    board = chess.Board()
    board.push_san("e4")
    assert (encode(board)[16] == 0.0).all()

def test_fullmove_plane():
    t = encode(chess.Board())
    expected = 1.0 / 500.0
    assert abs(t[17, 0, 0].item() - expected) < 1e-6
    assert (t[17] == t[17, 0, 0]).all()

def test_encode_flip_mirrors_king():
    from board_encoder import encode_flip
    from move_encoder import move_to_index
    board = chess.Board()
    idx = move_to_index(chess.Move.from_uci("e2e4"), board)
    t, _ = encode_flip(board, idx)
    assert t.shape == (18, 8, 8)
    assert t[5, 0, 3] == 1.0  # white king e1=(0,4) -> after flip -> (0,3)
    assert t[5, 0, 4] == 0.0

def test_encode_flip_swaps_castling():
    from board_encoder import encode_flip
    from move_encoder import move_to_index
    board = chess.Board()
    idx = move_to_index(chess.Move.from_uci("e2e4"), board)
    t, _ = encode_flip(board, idx)
    assert t[12, 0, 0] == 1.0 and t[12, 0, 7] == 1.0

def test_encode_flip_idx_in_range():
    from board_encoder import encode_flip
    from move_encoder import move_to_index
    board = chess.Board()
    for move in board.legal_moves:
        _, flipped = encode_flip(board, move_to_index(move, board))
        assert 0 <= flipped < 4672
