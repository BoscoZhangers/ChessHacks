import numpy as np
import torch
import chess
from .board_state import board_to_tensor
from .config import NUM_SIMULATIONS, C_PUCT
from .network_architecture import network_output_to_move_probs


class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior_prob=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        
        self.children = {}  # move -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob  # From policy network
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        """Average value from all visits"""
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count


class MCTS:
    def __init__(self, network, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT, device=None):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # Exploration constant
        self.device = device if device else torch.device('cpu')
        
    def search(self, board):
        """
        Run MCTS from this position, return best move
        """
        root = MCTSNode(board)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # PHASE 1: SELECT - tree walk until we hit a leaf
            while not node.is_leaf() and not node.board.is_game_over():
                node = self._select_child(node)
                search_path.append(node)
            
            # PHASE 2: EXPAND - if not terminal, add children
            value = 0
            if not node.board.is_game_over():
                value = self._expand_node(node)
            else:
                # Terminal node - get actual outcome
                result = node.board.result()
                value = self._result_to_value(result, board.turn)
            
            # PHASE 4: BACKPROPAGATE - update all nodes in path
            self._backpropagate(search_path, value)
        
        # Return move with most visits (most promising after search)
        return self._best_move(root)
    
    def _select_child(self, node):
        """
        Select child with highest UCB score
        UCB balances: exploitation (high value) + exploration (less visited)
        """
        best_score = -float('inf')
        best_child = None
        
        for move, child in node.children.items():
            Q = child.value()  # Exploitation: average value
            U = (self.c_puct * child.prior_prob * 
                 np.sqrt(node.visit_count) / (1 + child.visit_count))  # Exploration
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def _result_to_value(self, result, turn):
        """
        Convert a game result string ('1-0', '0-1', '1/2-1/2') 
        to a value from the perspective of 'turn' (True=White, False=Black).
        """
        if result == "1-0":
            return 1 if turn else -1
        elif result == "0-1":
            return -1 if turn else 1
        else:  # draw "1/2-1/2" or "*"
            return 0

    def _expand_node(self, node):
        """
        Add all legal moves as children, using network for priors and value.
        """
        board_tensor = board_to_tensor(node.board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).float()  # Shape: (1,13,8,8)
        board_tensor = board_tensor.to(self.device)
        self.network.to(self.device)  # Ensure network is on the same device

        with torch.no_grad():
            policy_output, value_output = self.network(board_tensor)
        
        policy_probs = policy_output[0].cpu().numpy()  # Shape: (4096,)
        position_value = value_output[0].item()        # Single number
        move_probs = network_output_to_move_probs(node.board, policy_probs)
        
        for move, prior_prob in move_probs.items():
            child_board = node.board.copy()
            child_board.push(move)
            
            child_node = MCTSNode(
                board=child_board,
                parent=node,
                move=move,
                prior_prob=prior_prob
            )
            node.children[move] = child_node
        
        return position_value  # Return value for backpropagation

    def _backpropagate(self, search_path, value):
        """
        Update visit counts and values for all nodes in path
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip for parent's perspective

    def _best_move(self, root):
        """
        Return move with highest visit count (most explored = most promising)
        """
        best_move = None
        best_visits = -1
        
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        
        return best_move

