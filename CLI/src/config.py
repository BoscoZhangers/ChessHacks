# config.py - Configuration for chess engine

# MCTS Configuration
NUM_SIMULATIONS = 100        # Number of MCTS simulations per move
C_PUCT = 1.5                 # Exploration constant for UCB

# Network Configuration
INPUT_CHANNELS = 13          # Number of input planes (6 own + 6 opponent + 1 metadata)
RESIDUAL_BLOCKS = 5          # Number of ResNet blocks
HIDDEN_CHANNELS = 128        # Channels in hidden layers
POLICY_OUTPUT_SIZE = 4096    # 64 * 64 possible moves

# Training Configuration
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

# Model Path
import os

# Resolve model path relative to this file so it's correct whether
# the package is imported from the repo root or when the process
# current working directory is different (avoids fragile ".." paths).
MODEL_PATH = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "models", "trained_model.pth")
)