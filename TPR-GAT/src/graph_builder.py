"""
graph_builder.py
Converts clip DataFrames into PyTorch Geometric graph objects.

- build_graph()         : 22 players -> PyG Data object
- build_context_vector(): 7 game-state features per clip
- compute_team_history(): team historical tactic distributions
- build_pyg_dataset()  : full list of PyG Data objects
- split_and_load()     : train/val/test DataLoaders split by match
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import LabelEncoder

from src.config import (
    PITCH_LENGTH, PITCH_WIDTH, PROXIMITY_M,
    TACTIC_CLASSES, BATCH_SIZE, RANDOM_SEED
)


def compute_team_history(df_train: pd.DataFrame) -> dict:
    """
    Compute each team's tactic distribution from training clips only.
    Returns {team_name: {tactic: fraction}}.
    MUST be computed from training data only — never from val/test.
    """
    history = {}
    for team, group in df_train.groupby("team"):
        counts = group["tactic_label"].value_counts(normalize=True)
        history[team] = {
            "high_press":     float(counts.get("high_press",    0.0)),
            "counter_attack": float(counts.get("counter_attack",0.0)),
            "build_up":       float(counts.get("build_up",      0.0)),
            "low_block":      float(counts.get("low_block",     0.0)),
        }
    return history


def build_context_vector(row, team_history: dict) -> torch.Tensor:
    """
    Build a 7-dimensional context tensor for one clip.
    Features: [score_norm, minute_norm, is_home,
               hist_high_press, hist_counter, hist_build_up, hist_low_block]
    """
    team = row["team"]
    hist = team_history.get(team, {
        "high_press": 0.25, "counter_attack": 0.25,
        "build_up":   0.25, "low_block":      0.25
    })
    return torch.tensor([
        float(np.clip(row["score_differential"], -3, 3)) / 3.0,
        float(row["match_minute"]) / 90.0,
        float(row["team_is_home"]),
        hist["high_press"],
        hist["counter_attack"],
        hist["build_up"],
        hist["low_block"],
    ], dtype=torch.float)


def build_graph(positions: list,
                proximity_m: float = PROXIMITY_M) -> Data:
    """
    Convert 22 player position dicts into a PyTorch Geometric Data object.

    Node features (5 per player):
        x_norm      : x / PITCH_LENGTH
        y_norm      : y / PITCH_WIDTH
        vx_norm     : vx / 10.0   (max sprint speed ~10 m/s)
        vy_norm     : vy / 10.0
        team_flag   : 1.0 = labeled team, 0.0 = opponent

    Edges: undirected, connect players within proximity_m metres.
    Fallback: if no pair within range, connect each node to nearest neighbour.
    """
    node_features = [[
        p["x"]  / PITCH_LENGTH,
        p["y"]  / PITCH_WIDTH,
        p["vx"] / 10.0,
        p["vy"] / 10.0,
        p["team_flag"]
    ] for p in positions]

    x = torch.tensor(node_features, dtype=torch.float)  # (22, 5)

    n    = len(positions)
    esrc, edst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i]["x"] - positions[j]["x"]
            dy = positions[i]["y"] - positions[j]["y"]
            if np.sqrt(dx*dx + dy*dy) <= proximity_m:
                esrc.append(i); edst.append(j)
                esrc.append(j); edst.append(i)

    # Fallback: no edges found — connect each node to nearest neighbour
    if not esrc:
        for i in range(n):
            best_j, best_d = (i+1) % n, float("inf")
            for j in range(n):
                if i == j:
                    continue
                dx = positions[i]["x"] - positions[j]["x"]
                dy = positions[i]["y"] - positions[j]["y"]
                d  = dx*dx + dy*dy
                if d < best_d:
                    best_d, best_j = d, j
            esrc.append(i); edst.append(best_j)
            esrc.append(best_j); edst.append(i)

    edge_index = torch.tensor([esrc, edst], dtype=torch.long)  # (2, num_edges)
    return Data(x=x, edge_index=edge_index)


def build_pyg_dataset(df_clips: pd.DataFrame,
                      team_history: dict,
                      label_encoder: LabelEncoder) -> list:
    """
    Convert every row of df_clips into a PyG Data object.
    Each object carries: x, edge_index, context, y_tactic, y_adapt, y_suggest.
    """
    dataset = []
    skipped = 0

    for idx, row in df_clips.iterrows():
        positions = row.get("player_positions", None)
        if not isinstance(positions, list) or len(positions) < 2:
            skipped += 1
            continue
        try:
            graph = build_graph(positions)
        except Exception:
            skipped += 1
            continue

        try:
            y_tactic  = int(label_encoder.transform([row["tactic_label"]])[0])
            y_suggest = int(label_encoder.transform([row["suggestion_label"]])[0])
        except Exception:
            skipped += 1
            continue

        graph.context   = build_context_vector(row, team_history)
        graph.y_tactic  = torch.tensor([y_tactic],             dtype=torch.long)
        graph.y_adapt   = torch.tensor([int(row["adaptation_flag"])], dtype=torch.long)
        graph.y_suggest = torch.tensor([y_suggest],            dtype=torch.long)
        dataset.append(graph)

    print(f"PyG dataset: {len(dataset)} graphs  |  skipped: {skipped}")
    return dataset


def split_and_load(df_clips: pd.DataFrame,
                   pyg_dataset: list,
                   train_ids: set,
                   val_ids:   set,
                   test_ids:  set):
    """
    Split by match_id (never by clip) and build DataLoaders.
    Returns (train_loader, val_loader, test_loader,
             train_data, val_data, test_data).
    """
    mid_series = df_clips["match_id"].reset_index(drop=True)
    train_data, val_data, test_data = [], [], []

    for i, graph in enumerate(pyg_dataset):
        if i >= len(mid_series):
            break
        mid = mid_series.iloc[i]
        if mid in train_ids:    train_data.append(graph)
        elif mid in val_ids:    val_data.append(graph)
        else:                   test_data.append(graph)

    print(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = PyGDataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = PyGDataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data


if __name__ == "__main__":
    print("graph_builder.py — import and call build_pyg_dataset() from main.py")
