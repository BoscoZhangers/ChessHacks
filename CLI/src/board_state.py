import numpy as np
import chess

def board_to_tensor(board):
    """
    Convert chess.Board to neural network input
    Output shape: (13, 8, 8)  -> channels-first for PyTorch
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            plane_idx = piece_to_plane[piece.piece_type]
            
            if piece.color == board.turn:
                tensor[plane_idx, rank, file] = 1.0
            else:
                tensor[plane_idx + 6, rank, file] = 1.0
    
    # Plane 12: metadata (castling, en passant, etc.)
    if board.has_kingside_castling_rights(board.turn):
        tensor[12, :, :] = 0.1
    if board.has_queenside_castling_rights(board.turn):
        tensor[12, :, :] += 0.1
    
    return tensor
