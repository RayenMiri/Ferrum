# board_encoder.py
import chess
import torch

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]

_QUEEN_FLIP_DIR = [0, 7, 6, 5, 4, 3, 2, 1]
_KNIGHT_FLIP = [57, 56, 59, 58, 61, 60, 63, 62]


def encode(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(18, 8, 8, dtype=torch.float32)
    for i, pt in enumerate(_PIECE_TYPES):
        for sq in board.pieces(pt, chess.WHITE):
            planes[i, sq // 8, sq % 8] = 1.0
        for sq in board.pieces(pt, chess.BLACK):
            planes[6 + i, sq // 8, sq % 8] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[12, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[12, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[12, 7, 0] = 1.0
    if board.ep_square is not None:
        planes[13, board.ep_square // 8, board.ep_square % 8] = 1.0
    planes[14] = board.halfmove_clock / 100.0
    rep = 2 if board.is_repetition(2) else (1 if board.is_repetition(1) else 0)
    planes[15] = float(rep)
    planes[16] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[17] = board.fullmove_number / 500.0
    return planes


def encode_flip(board: chess.Board, move_idx: int) -> tuple:
    """Return (horizontally-mirrored tensor, mirrored move index)."""
    return encode(board).flip(2), _flip_move_index(move_idx)


def _flip_move_index(move_idx: int) -> int:
    plane = move_idx // 64
    sq = move_idx % 64
    new_sq = (sq // 8) * 8 + (7 - sq % 8)
    if plane < 56:
        new_plane = _QUEEN_FLIP_DIR[plane // 7] * 7 + plane % 7
    elif plane < 64:
        new_plane = _KNIGHT_FLIP[plane - 56]
    else:
        offset = plane - 64
        new_direction = [2, 1, 0][offset % 3]
        new_plane = 64 + (offset // 3) * 3 + new_direction
    return new_plane * 64 + new_sq
