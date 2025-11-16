from .utils import chess_manager, GameContext
from chess import Move
import random
import time
from .chess_engine import ChessEngine


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
engine = ChessEngine()

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1

    best_move, move_stats = engine.get_move_with_stats(ctx.board)
    # move_probs = {
    #     move: weight / total_weight
    #     for move, weight in zip(legal_moves, move_weights)
    # }
    # ctx.logProbabilities(move_probs)

    move_probs = {
        move: move_stat_dict['prior']
        for move, move_stat_dict in move_stats.items()
    }
    ctx.logProbabilities(move_probs)
    return best_move
    #return engine.get_best_move(ctx.board)
    # return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
