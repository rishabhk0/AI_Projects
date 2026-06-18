"""
evaluate.py — Test evaluation, all result plots, and attention visualisation.

Public API
----------
run_evaluation(model, test_loader, test_data, le, class_weights_tensor, history)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Batch
from mplsoccer import Pitch

from config import DEVICE, CHECKPOINT_PATH
from model import GATWithAttention
from train import evaluate_model


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_test_results(test_m, le):
    print("=" * 60)
    print("FINAL TEST RESULTS — WITH CLASS IMBALANCE FIXES")
    print("=" * 60)
    print(f"\nHEAD 1 — Tactic Classifier")
    print(f"  Accuracy   : {test_m['tactic_acc']*100:.1f}%")
    print(f"  Macro F1   : {test_m['tactic_f1']:.3f}")
    print(f"\nHEAD 2 — Adaptation Flag")
    print(f"  Accuracy   : {test_m['adapt_acc']*100:.1f}%")
    print(f"  Macro F1   : {test_m['adapt_f1']:.3f}")
    print(f"\nHEAD 3 — Suggestion Engine")
    print(f"  Accuracy   : {test_m['suggest_acc']*100:.1f}%")
    print(f"  Macro F1   : {test_m['suggest_f1']:.3f}")
    print(f"\nPer-class F1 — HEAD 1:")
    print(f"  build_up        : {test_m['f1_build_up']:.3f}   (baseline: 0.930)")
    print(f"  counter_attack  : {test_m['f1_counter_attack']:.3f}   (baseline: 0.000)")
    print(f"  high_press      : {test_m['f1_high_press']:.3f}   (baseline: 0.000)")
    print(f"  low_block       : {test_m['f1_low_block']:.3f}   (baseline: 0.860)")
    print(f"  Macro avg       : {test_m['tactic_f1']:.3f}   (baseline: 0.447)")
    print(f"\nFull classification report (HEAD 1):")
    print(classification_report(
        test_m["trues"]["t"], test_m["preds"]["t"],
        labels=[0, 1, 2, 3], target_names=le.classes_, zero_division=0,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(test_m, le,
                             out="results/confusion_matrices.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set (After Class Imbalance Fixes)",
                 fontsize=12, fontweight="bold")

    configs = [
        (test_m["preds"]["t"], test_m["trues"]["t"],
         le.classes_.tolist(),         "HEAD 1 — Tactic",     "#1F6B3A"),
        (test_m["preds"]["a"], test_m["trues"]["a"],
         ["No adapt", "Adapting"],     "HEAD 2 — Adaptation", "#1565C0"),
        (test_m["preds"]["s"], test_m["trues"]["s"],
         le.classes_.tolist(),         "HEAD 3 — Suggestion", "#E65100"),
    ]
    for ax, (pred, true, names, title, color) in zip(axes, configs):
        cm      = confusion_matrix(true, pred)
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
        sns.heatmap(cm_norm, ax=ax, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=names, yticklabels=names,
                    linewidths=0.5, linecolor="white",
                    vmin=0, vmax=1, cbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HISTORY + OVERFITTING CHECK
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, test_m,
                           out="results/training_history.png"):
    import pandas as pd
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training History — Class-Weighted TacticalGAT",
                 fontsize=13, fontweight="bold")

    # Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=2, color="#1565C0")
    axes[0, 0].plot(df["epoch"], df["loss"],       label="Val",   linewidth=2, color="#E65100", linestyle="--")
    axes[0, 0].set_title("Loss — divergence = overfitting", fontsize=10)
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")

    # Val tactic accuracy
    axes[0, 1].plot(df["epoch"], df["tactic_acc"], linewidth=2, color="#1F6B3A")
    axes[0, 1].set_title("Val Tactic Accuracy", fontsize=10)
    axes[0, 1].grid(alpha=0.3); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)

    # Per-class F1
    axes[1, 0].plot(df["epoch"], df["f1_build_up"],       label="build_up",      color="#1F6B3A", linewidth=1.5)
    axes[1, 0].plot(df["epoch"], df["f1_low_block"],      label="low_block",     color="#1565C0", linewidth=1.5)
    axes[1, 0].plot(df["epoch"], df["f1_counter_attack"], label="counter_attack",color="#E65100", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["f1_high_press"],     label="high_press",    color="#6A1B9A", linewidth=2)
    axes[1, 0].set_title("Per-class F1 on validation set", fontsize=10)
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylim(0, 1)

    # Macro F1 + test line
    axes[1, 1].plot(df["epoch"], df["tactic_f1"], label="Val Macro F1",
                    color="#8E24AA", linewidth=2)
    axes[1, 1].axhline(y=test_m["tactic_f1"], color="red", linestyle=":",
                        linewidth=1.5, label=f"Test F1 = {test_m['tactic_f1']:.3f}")
    axes[1, 1].set_title("Val Macro F1 — should plateau, not drop", fontsize=10)
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Overfitting summary
    best_epoch = df.loc[df["loss"].idxmin(), "epoch"]
    best_f1    = df.loc[df["loss"].idxmin(), "tactic_f1"]
    final_f1   = df["tactic_f1"].iloc[-1]
    print(f"\nBest val loss at epoch: {best_epoch}")
    print(f"Val macro F1 at best checkpoint: {best_f1:.3f}")
    print(f"Val macro F1 at final epoch:     {final_f1:.3f}")
    print(f"Test macro F1:                   {test_m['tactic_f1']:.3f}")
    if final_f1 < best_f1 - 0.05:
        print("WARNING: Val F1 dropped significantly — possible overfitting after checkpoint.")
    else:
        print("Val F1 stable. No clear overfitting signal.")


# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION VISUALISATION ON PITCH
# ══════════════════════════════════════════════════════════════════════════════

def plot_attention_on_pitch(model, test_data, le,
                             out="results/attention_pitch.png"):
    att_model = GATWithAttention(model).to(DEVICE)
    att_model.eval()

    # Find one example per predicted class
    examples = {}
    for g in test_data:
        gb = Batch.from_data_list([g]).to(DEVICE)
        with torch.no_grad():
            o1, _, _, _, _ = att_model(gb)
        pred = le.inverse_transform([o1.argmax().item()])[0]
        if pred not in examples:
            examples[pred] = g
        if len(examples) >= 4:
            break

    tactic_colors = {
        "high_press":     "#E65100",
        "counter_attack": "#1565C0",
        "build_up":       "#1F6B3A",
        "low_block":      "#6A1B9A",
    }

    n = len(examples)
    if n == 0:
        print("No examples found for attention visualisation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("GAT Attention Weights on Pitch — One Example per Predicted Tactic",
                 fontsize=13, fontweight="bold")

    for ax, (tactic, g) in zip(axes, examples.items()):
        gb = Batch.from_data_list([g]).to(DEVICE)
        with torch.no_grad():
            o1, _, _, att_ei, att_w = att_model(gb)

        positions = g.x.numpy()
        src = att_ei[0].cpu().numpy()
        dst = att_ei[1].cpu().numpy()
        w   = att_w.mean(dim=1).cpu().numpy()
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)

        pitch = Pitch(pitch_color="grass", line_color="white",
                      pitch_type="statsbomb")
        pitch.draw(ax=ax)

        for s, d, wn in zip(src, dst, w_norm):
            if s >= d: continue
            x_s = positions[s, 0] * 120; y_s = positions[s, 1] * 80
            x_d = positions[d, 0] * 120; y_d = positions[d, 1] * 80
            ax.plot([x_s, x_d], [y_s, y_d],
                    color=tactic_colors.get(tactic, "white"),
                    alpha=float(wn) * 0.8 + 0.05,
                    linewidth=float(wn) * 3 + 0.3)

        team_flag = positions[:, 4]
        for i, (xn, yn) in enumerate(zip(positions[:, 0] * 120,
                                          positions[:, 1] * 80)):
            ax.scatter(xn, yn, s=120,
                       c="#FFFFFF" if team_flag[i] == 1 else "#AAAAAA",
                       zorder=5, edgecolors="black", linewidths=0.8)

        pred_cls = le.inverse_transform([o1.argmax().item()])[0]
        ax.set_title(f"Predicted: {pred_cls}", fontsize=11, fontweight="bold",
                     color=tactic_colors.get(tactic, "white"))

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model, test_loader, test_data, le,
                   class_weights_tensor, history):
    """Load best checkpoint, evaluate, save all figures."""
    os.makedirs("results", exist_ok=True)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_m = evaluate_model(model, test_loader, DEVICE, class_weights_tensor)

    print_test_results(test_m, le)
    plot_confusion_matrices(test_m, le)
    plot_training_history(history, test_m)
    plot_attention_on_pitch(model, test_data, le)

    return test_m
