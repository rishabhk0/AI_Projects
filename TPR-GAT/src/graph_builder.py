"""
graph_builder.py — Build PyTorch Geometric graphs, encode labels,
compute class weights, and create DataLoaders.

Public API
----------
split_by_match(df_clips)                                 -> (train_ids, val_ids, test_ids)
encode_labels(df_clips)                                  -> (df_clips, le)
compute_class_weights(df_clips, train_match_ids, le)     -> Tensor
build_pyg_dataset(df_clips, train_match_ids)             -> list[Data]
make_dataloaders(pyg_dataset, train_ids, val_ids)        -> (train_loader, val_loader, test_loader,
                                                             train_data, val_data, test_data)
"""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import (
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO,
    TACTIC_CLASSES, PROXIMITY_M,
    PITCH_LENGTH, PITCH_WIDTH,
    BATCH_SIZE, DEVICE,
    NODE_FEATURES, CONTEXT_FEATURES,
)


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_by_match(df_clips):
    """
    Split match IDs into train / val / test by ratio.
    Split is on matches (not clips) to prevent data leakage.
    """
    all_ids = df_clips["match_id"].unique().copy()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_ids)

    n       = len(all_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_ids = set(all_ids[:n_train])
    val_ids   = set(all_ids[n_train:n_train + n_val])
    test_ids  = set(all_ids[n_train + n_val:])

    print(f"Match split — train: {len(train_ids)} | "
          f"val: {len(val_ids)} | test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# ══════════════════════════════════════════════════════════════════════════════
# LABEL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode_labels(df_clips):
    """
    Fit LabelEncoder on TACTIC_CLASSES and add integer columns.
    Encoding: build_up=0, counter_attack=1, high_press=2, low_block=3
    Returns (df_clips_with_columns, le).
    """
    le = LabelEncoder().fit(TACTIC_CLASSES)
    df = df_clips.copy()
    df["tactic_label_enc"]      = le.transform(df["tactic_label"])
    df["suggestion_label_enc"]  = le.transform(df["suggestion_label"])

    print("Label encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} -> {cls}")
    return df, le


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(df_clips, train_match_ids, le) -> torch.Tensor:
    """
    Inverse-frequency weights from training set only.
    weight_i = n_train_total / (n_classes * count_i)
    high_press gets ~4000x the weight of build_up.
    """
    train_counts  = (df_clips[df_clips["match_id"].isin(train_match_ids)]
                     ["tactic_label"].value_counts())
    n_total = train_counts.sum()
    n_cls   = len(TACTIC_CLASSES)

    weights = []
    print("Class weights for cross-entropy (HEAD 1):")
    for cls in le.classes_:
        count  = train_counts.get(cls, 1)
        weight = n_total / (n_cls * count)
        weights.append(weight)
        print(f"  {cls:<20}  count={count:>7,}   weight={weight:>9.2f}")

    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph(positions, proximity_m=PROXIMITY_M) -> Data:
    """Convert 22 player dicts into a PyG Data object (nodes + edges only)."""
    node_features = [[
        p["x"]  / PITCH_LENGTH,
        p["y"]  / PITCH_WIDTH,
        p["vx"] / 10.0,
        p["vy"] / 10.0,
        p["team_flag"],
    ] for p in positions]

    x = torch.tensor(node_features, dtype=torch.float)
    n = len(positions)
    esrc, edst = [], []

    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i]["x"] - positions[j]["x"]
            dy = positions[i]["y"] - positions[j]["y"]
            if np.sqrt(dx * dx + dy * dy) <= proximity_m:
                esrc.append(i); edst.append(j)
                esrc.append(j); edst.append(i)

    if not esrc:
        # Fallback: nearest neighbour so graph is never empty
        for i in range(n):
            best_j, best_d = (i + 1) % n, float("inf")
            for j in range(n):
                if i == j: continue
                d = (positions[i]["x"] - positions[j]["x"]) ** 2 + \
                    (positions[i]["y"] - positions[j]["y"]) ** 2
                if d < best_d:
                    best_d, best_j = d, j
            esrc.append(i); edst.append(best_j)
            esrc.append(best_j); edst.append(i)

    return Data(x=x, edge_index=torch.tensor([esrc, edst], dtype=torch.long))


def _build_context_vector(row, history: dict) -> torch.Tensor:
    """7-dimensional context vector for one clip."""
    team = row["team"]
    hist = history.get(team, {
        "high_press": 0.25, "counter_attack": 0.25,
        "build_up": 0.25,   "low_block": 0.25,
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


def build_pyg_dataset(df_clips, train_match_ids) -> list:
    """
    Build one PyG Data object per clip row.
    Context history is computed from training set only.
    """
    # Team tactic history — training clips only
    df_train = df_clips[df_clips["match_id"].isin(train_match_ids)]
    history  = {}
    for team, grp in df_train.groupby("team"):
        counts = grp["tactic_label"].value_counts(normalize=True)
        history[team] = {cls: float(counts.get(cls, 0.0)) for cls in TACTIC_CLASSES}

    dataset, skipped = [], 0

    for idx, row in df_clips.iterrows():
        pos = row.get("player_positions")
        if not isinstance(pos, list) or len(pos) < 22:
            skipped += 1
            continue

        graph   = _build_graph(pos, proximity_m=PROXIMITY_M)
        context = _build_context_vector(row, history)

        graph.context   = context.unsqueeze(0)
        graph.y_tactic  = torch.tensor([int(row["tactic_label_enc"])],   dtype=torch.long)
        graph.y_adapt   = torch.tensor([int(row["adaptation_flag"])],     dtype=torch.long)
        graph.y_suggest = torch.tensor([int(row["suggestion_label_enc"])], dtype=torch.long)
        graph.match_id  = int(row["match_id"])

        dataset.append(graph)
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1:,} graphs built")

    print(f"PyG dataset: {len(dataset)} graphs | skipped: {skipped}")
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
# DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

def make_dataloaders(pyg_dataset, train_match_ids, val_match_ids):
    """Split pyg_dataset by match_id and return DataLoaders + raw lists."""
    train_data, val_data, test_data = [], [], []
    for g in pyg_dataset:
        mid = g.match_id
        if mid in train_match_ids:
            train_data.append(g)
        elif mid in val_match_ids:
            val_data.append(g)
        else:
            test_data.append(g)

    print(f"Clips — train: {len(train_data):,} | "
          f"val: {len(val_data):,} | test: {len(test_data):,}")

    train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = PyGDataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = PyGDataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data
