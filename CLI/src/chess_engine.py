import torch
import chess
from .network_architecture import ChessNetwork
from .MCTSAlgorithm import MCTS, MCTSNode
from .config import MODEL_PATH, NUM_SIMULATIONS
import os

class ChessEngine:
    """
    Main chess engine class that combines neural network and MCTS.
    """
    
    def __init__(self, model_path=MODEL_PATH, num_simulations=NUM_SIMULATIONS, device=None):
        """
        Initialize the chess engine.
        
        Args:
            model_path: Path to trained model weights
            num_simulations: Number of MCTS simulations per move
            device: torch device (cuda/cpu), auto-detected if None
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize network
        self.network = ChessNetwork()
        
        # Load trained weights if available
        if os.path.exists(model_path):
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
                print("Using randomly initialized network")
        else:
            print(f"Model file {model_path} not found. Using randomly initialized network")
        
        self.network.to(self.device)
        self.network.eval()  # Set to evaluation mode
        
        # Initialize MCTS
        self.mcts = MCTS(self.network, num_simulations=num_simulations, device=self.device)
        
    def get_best_move(self, board, info_callback=None):
        """
        Get the best move for the given position.
        
        Args:
            board: chess.Board object
            info_callback: Optional callback function to send info strings
            
        Returns:
            chess.Move object or None if no legal moves
        """
        if board.is_game_over():
            return None
        
        if info_callback:
            info_callback(f"Starting MCTS with {self.mcts.num_simulations} simulations")
        
        best_move = self.mcts.search(board)
        
        if info_callback and best_move:
            info_callback(f"Best move selected: {best_move.uci()}")
        
        return best_move
    
    def get_move_with_stats(self, board):
        """
        Get best move along with statistics from MCTS.
        
        Returns:
            dict with keys: 'move', 'visit_counts', 'values'
        """
        root = MCTSNode(board)
        
        # Run MCTS
        for _ in range(self.mcts.num_simulations):
            node = root
            search_path = [node]
            
            while not node.is_leaf() and not node.board.is_game_over():
                node = self.mcts._select_child(node)
                search_path.append(node)
            
            value = 0
            if not node.board.is_game_over():
                value = self.mcts._expand_node(node)
            else:
                result = node.board.result()
                value = self.mcts._result_to_value(result, board.turn)
            
            self.mcts._backpropagate(search_path, value)
        
        # Collect statistics
        move_stats = {}
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            move_stats[move.uci()] = {
                'visits': child.visit_count,
                'value': child.value(),
                'prior': child.prior_prob
            }
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        
        return best_move, move_stats
