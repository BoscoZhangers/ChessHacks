import torch
import torch.nn as nn
from .config import INPUT_CHANNELS, RESIDUAL_BLOCKS, HIDDEN_CHANNELS, POLICY_OUTPUT_SIZE

class ChessNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared backbone
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(HIDDEN_CHANNELS)
        self.conv3 = nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(HIDDEN_CHANNELS)
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_CHANNELS) for _ in range(RESIDUAL_BLOCKS)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_OUTPUT_SIZE)

        
        # Value head
        self.value_conv = nn.Conv2d(HIDDEN_CHANNELS, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_fc1_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        # Shared processing
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.flatten(1)
        policy = self.policy_fc(policy)
        # NOTE: No softmax here — use raw logits for CrossEntropyLoss

        # Value head
        value = torch.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1_bn(self.value_fc1(value)))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out


def network_output_to_move_probs(board, policy_output):
    """
    Convert raw network output to probability distribution over legal moves
    """
    legal_moves = list(board.legal_moves)
    move_to_idx = {}
    
    # Map each legal move to its index in the 4096 output
    for move in legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        move_idx = from_square * 64 + to_square
        move_to_idx[move] = move_idx
    
    # Extract probabilities for legal moves only
    legal_move_probs = {}
    total_prob = 0.0
    
    for move, idx in move_to_idx.items():
        prob = policy_output[idx]
        legal_move_probs[move] = prob
        total_prob += prob
    
    # Renormalize (since we filtered illegal moves)
    for move in legal_move_probs:
        legal_move_probs[move] /= total_prob
    
    return legal_move_probs