import math
import torch
import torch.nn.functional as F
import chess
import chess.engine

from move_encoder import index_to_move, legal_move_mask
from board_encoder import encode


def compute_perplexity(model, val_loader, device) -> float:
    training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0

    try:
        with torch.no_grad():
            for batch in val_loader:
                state, policy_target, _, legal_mask = batch
                state = state.to(device)
                policy_target = policy_target.to(device)
                legal_mask = legal_mask.to(device)

                logits, _ = model(state)
                logits = logits.masked_fill(~legal_mask, -torch.inf)
                loss = F.cross_entropy(logits, policy_target, reduction="sum")
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite perplexity loss — check for all-illegal masks or corrupt policy targets"
                    )
                total_loss += loss.item()
                total_samples += policy_target.size(0)
    finally:
        model.train(training)

    avg_loss = total_loss / max(1, total_samples)
    return math.exp(avg_loss)


def probe_elo_vs_stockfish(model, stockfish_path: str, num_games: int = 20,
                            depth: int = 1, device: str = "cpu") -> dict:
    model.eval()
    wins = 0
    draws = 0
    losses = 0

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    fallback_count = 0

    try:
        for game_idx in range(num_games):
            board = chess.Board()
            ferrum_is_white = (game_idx % 2 == 0)

            while not board.is_game_over():
                ferrum_turn = (board.turn == chess.WHITE) == ferrum_is_white

                if ferrum_turn:
                    mask = legal_move_mask(board).to(device)
                    state = encode(board).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = model(state)
                    logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
                    probs = torch.softmax(logits, dim=-1)
                    idx = torch.multinomial(probs, num_samples=1).item()
                    move = index_to_move(idx, board)
                    if move not in board.legal_moves:
                        fallback_count += 1
                        move = list(board.legal_moves)[0]
                    board.push(move)
                else:
                    result = engine.play(board, chess.engine.Limit(depth=depth))
                    if result.move is None:
                        break
                    board.push(result.move)

            outcome = board.outcome()
            if outcome is None or outcome.winner is None:
                draws += 1
            elif (outcome.winner == chess.WHITE) == ferrum_is_white:
                wins += 1
            else:
                losses += 1
    finally:
        engine.quit()

    score = (wins + 0.5 * draws) / num_games
    return {"wins": wins, "draws": draws, "losses": losses,
            "score": score, "fallback_moves": fallback_count}
