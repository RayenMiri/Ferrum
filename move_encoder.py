# move_encoder.py
import chess
import torch

_QUEEN_DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
_KNIGHT_OFFSETS = [(2,1),(2,-1),(1,2),(1,-2),(-1,2),(-1,-2),(-2,1),(-2,-1)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    from_sq = move.from_square
    to_sq   = move.to_square
    from_row, from_col = from_sq // 8, from_sq % 8
    dr = to_sq // 8 - from_row
    dc = to_sq % 8  - from_col

    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
        dir_idx = 0 if dc < 0 else (1 if dc == 0 else 2)
        return (64 + piece_idx * 3 + dir_idx) * 64 + from_row * 8 + from_col

    for k_idx, (kdr, kdc) in enumerate(_KNIGHT_OFFSETS):
        if dr == kdr and dc == kdc:
            return (56 + k_idx) * 64 + from_row * 8 + from_col

    for dir_idx, (ddr, ddc) in enumerate(_QUEEN_DIRS):
        if _sign(dr) == ddr and _sign(dc) == ddc:
            dist = max(abs(dr), abs(dc))
            if 1 <= dist <= 7 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                return (dir_idx * 7 + dist - 1) * 64 + from_row * 8 + from_col

    raise ValueError(f"Cannot encode move {move}: dr={dr}, dc={dc}")


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    plane    = idx // 64
    sq_flat  = idx % 64
    from_row, from_col = sq_flat // 8, sq_flat % 8
    from_sq  = from_row * 8 + from_col

    if plane < 56:
        dir_idx = plane // 7
        dist    = plane % 7 + 1
        ddr, ddc = _QUEEN_DIRS[dir_idx]
        to_sq = (from_row + ddr * dist) * 8 + (from_col + ddc * dist)
        piece = board.piece_at(from_sq)
        if (piece and piece.piece_type == chess.PAWN
                and ((piece.color == chess.WHITE and (to_sq // 8) == 7)
                     or (piece.color == chess.BLACK and (to_sq // 8) == 0))):
            return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)

    elif plane < 64:
        ddr, ddc = _KNIGHT_OFFSETS[plane - 56]
        return chess.Move(from_sq, (from_row + ddr) * 8 + (from_col + ddc))

    else:
        offset     = plane - 64
        piece_type = [chess.KNIGHT, chess.BISHOP, chess.ROOK][offset // 3]
        dc         = [-1, 0, 1][offset % 3]
        piece      = board.piece_at(from_sq)
        dr         = 1 if (piece and piece.color == chess.WHITE) else -1
        return chess.Move(from_sq, (from_row + dr) * 8 + (from_col + dc),
                          promotion=piece_type)


def legal_move_mask(board: chess.Board) -> torch.BoolTensor:
    mask = torch.zeros(4672, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
