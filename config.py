import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
FILTERS = 128
NUM_BLOCKS = 10
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
MAX_EPOCHS = 50
VALUE_LOSS_WEIGHT = 0.0
CHECKPOINT_STEPS = 1000
NUM_MCTS_SIMS = 200
ELO_MIN = 1500
ELO_MAX = 2200

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RUNS_DIR = BASE_DIR / "runs"

for _d in (DATA_DIR, CHECKPOINT_DIR, RUNS_DIR):
    _d.mkdir(exist_ok=True)
