"""
config.py — All constants and hyperparameters for TacticalGAT.
Change values here; every other module imports from this file.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ── Pitch ─────────────────────────────────────────────────────────────────────
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0
HALFWAY_X    = 52.5   # StatsBomb x-coordinate of halfway line (120-unit scale)

# ── Sliding window ────────────────────────────────────────────────────────────
WINDOW_SEC    = 15
FPS           = 25
WINDOW_FRAMES = WINDOW_SEC * FPS

# ── Graph construction ────────────────────────────────────────────────────────
# Updated from ablation study: 25 m outperforms 15 m (+2.3 pp tactic accuracy)
PROXIMITY_M = 25.0

# ── Tactic classes ────────────────────────────────────────────────────────────
# LabelEncoder order is alphabetical: build_up=0, counter_attack=1,
# high_press=2, low_block=3
TACTIC_CLASSES = ["build_up", "counter_attack", "high_press", "low_block"]

# ── Competitions ──────────────────────────────────────────────────────────────
TARGET_COMPETITIONS = [
    {"competition_id": 11,  "season_id": 27,  "name": "La Liga 2015/16"},
    {"competition_id": 223, "season_id": 282, "name": "Copa America 2024"},
    {"competition_id": 43,  "season_id": 106, "name": "FIFA World Cup 2022"},
]

# ── Data split (by match, not by clip) ───────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

# ── Class imbalance fixes ─────────────────────────────────────────────────────
TARGET_PER_CLASS = 3_000   # oversample rare classes to this many training clips
BUILD_UP_CAP     = 30_000  # cap build_up in training to reduce majority dominance

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
N_EPOCHS     = 20
LR           = 0.001
WEIGHT_DECAY = 1e-4

# ── Model architecture (best config from ablation) ────────────────────────────
NODE_FEATURES    = 5   # x_norm, y_norm, vx_norm, vy_norm, team_flag
CONTEXT_FEATURES = 7   # score_diff, minute, home_flag, tactic_history x4
HIDDEN_DIM       = 64
HEADS            = 8
N_CLASSES        = 4
DROPOUT          = 0.3
N_GAT_LAYERS     = 4   # updated from ablation (was 3, +19.3 pp)
# Activation: ReLU   (was ELU, +8.4 pp from ablation)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "results/best_model.pt"

# ── Suggestion lookup: (score_bracket, minute_bracket) -> tactic ──────────────
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
