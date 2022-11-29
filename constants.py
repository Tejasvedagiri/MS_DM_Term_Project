import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIMENSION = 200
PROBABILITY_DISTRIBUTION_DIMENSION = 20
NUM_EPOCHS = 50
BATCH_SIZE = 256
LR_RATE = 3e-4  # Karpathy constant\
MODEL_DIR = "model"