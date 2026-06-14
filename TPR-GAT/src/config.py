"""
config.py
All constants and hyperparameters in one place.
Change a value here and it flows through every other file automatically.
"""
import torch
import numpy as np

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ── Clip / window ────────────────────────────────────────────────────────────
WINDOW_SEC    = 15          # each training sample covers 15 seconds
FPS           = 25          # SoccerNet tracking frame rate
WINDOW_FRAMES = WINDOW_SEC * FPS   # = 375 frames per clip

# ── Pitch dimensions (metres) ─────────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
HALFWAY_X    = 52.5         # x-coordinate of the halfway line

# ── Graph construction ────────────────────────────────────────────────────────
PROXIMITY_M  = 15.0         # two players are connected if within this distance

# ── Tactic classes ────────────────────────────────────────────────────────────
# Order must stay consistent — LabelEncoder maps alphabetically
TACTIC_CLASSES = ["build_up", "counter_attack", "high_press", "low_block"]

# ── StatsBomb competitions ────────────────────────────────────────────────────
TARGET_COMPETITIONS = [
    {"competition_id": 11,  "season_id": 27,  "name": "La Liga 2015/16"},
    {"competition_id": 223, "season_id": 282, "name": "Copa America 2024"},
    {"competition_id": 43,  "season_id": 106, "name": "FIFA World Cup 2022"},
]

# ── Model hyperparameters ─────────────────────────────────────────────────────
NODE_FEATURES    = 5    # x_norm, y_norm, vx_norm, vy_norm, team_flag
CONTEXT_FEATURES = 7    # score, minute, home_flag, 4x tactic history
HIDDEN_DIM       = 64
GAT_HEADS        = 8
N_CLASSES        = 4
DROPOUT          = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
N_EPOCHS     = 15
LR           = 0.001
WEIGHT_DECAY = 1e-4
LR_PATIENCE  = 5
LR_FACTOR    = 0.5
GRAD_CLIP    = 1.0

# ── Multi-task loss weights ───────────────────────────────────────────────────
W_TACTIC  = 1.0   # HEAD 1 — primary task, gets full weight
W_ADAPT   = 0.5   # HEAD 2 — secondary
W_SUGGEST = 0.5   # HEAD 3 — secondary

# ── File paths ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "best_model.pt"
RESULTS_DIR     = "results"

# ── Suggestion lookup: (score_bracket, minute_bracket) -> recommended tactic ──
SUGGESTION_LOOKUP = {
    ("winning_2+", "early"): "build_up",
    ("winning_2+", "mid"):   "build_up",
    ("winning_2+", "late"):  "low_block",
    ("winning_1",  "early"): "build_up",
    ("winning_1",  "mid"):   "build_up",
    ("winning_1",  "late"):  "low_block",
    ("drawing",    "early"): "build_up",
    ("drawing",    "mid"):   "high_press",
    ("drawing",    "late"):  "high_press",
    ("losing_1",   "early"): "high_press",
    ("losing_1",   "mid"):   "high_press",
    ("losing_1",   "late"):  "counter_attack",
    ("losing_2+",  "early"): "high_press",
    ("losing_2+",  "mid"):   "counter_attack",
    ("losing_2+",  "late"):  "counter_attack",
}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    print(f"Device    : {DEVICE}")
    print(f"Classes   : {TACTIC_CLASSES}")
    print(f"Proximity : {PROXIMITY_M}m")
    print(f"Window    : {WINDOW_SEC}s = {WINDOW_FRAMES} frames")
    print(f"Comps     : {[c['name'] for c in TARGET_COMPETITIONS]}")
