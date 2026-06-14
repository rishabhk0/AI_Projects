"""
evaluate.py
Evaluation outputs: confusion matrices, training history plots,
GAT attention visualisation on a soccer pitch, results summary.

Run standalone:
    python src/evaluate.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    DEVICE, PITCH_LENGTH, PITCH_WIDTH, RESULTS_DIR, CHECKPOINT_PATH
)


def plot_confusion_matrices(test_m: dict, class_names: list, save: bool = True):
    """Three normalised confusion matrices, one per output head."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")

    configs = [
        (test_m["preds"]["t"], test_m["trues"]["t"],
         class_names,             "HEAD 1 — Tactic",     "#1F6B3A"),
        (test_m["preds"]["a"], test_m["trues"]["a"],
         ["No adapt","Adapting"], "HEAD 2 — Adaptation", "#1565C0"),
        (test_m["preds"]["s"], test_m["trues"]["s"],
         class_names,             "HEAD 3 — Suggestion", "#E65100"),
    ]

    for ax, (pred, true, names, title, color) in zip(axes, configs):
        cm      = confusion_matrix(true, pred)
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
        sns.heatmap(cm_norm, ax=ax, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=names, yticklabels=names,
                    linewidths=0.5, linecolor="white", vmin=0, vmax=1, cbar=False)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png",
                    dpi=130, bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR}/confusion_matrices.png")
    plt.show()


def plot_training_history(history: list, save: bool = True):
    """Loss and accuracy curves for all three heads over all epochs."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0,0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0,0].plot(df["epoch"], df["loss"],       label="Val")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(df["epoch"], df["tactic_acc"],  color="#1F6B3A")
    axes[0,1].set_title("HEAD 1 Tactic Accuracy"); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(df["epoch"], df["adapt_acc"],   color="#1565C0")
    axes[1,0].set_title("HEAD 2 Adaptation Accuracy"); axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(df["epoch"], df["suggest_acc"], color="#E65100")
    axes[1,1].set_title("HEAD 3 Suggestion Accuracy"); axes[1,1].grid(alpha=0.3)

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/training_history.png",
                    dpi=130, bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR}/training_history.png")
    plt.show()


# ── GAT attention visualisation ────────────────────────────────────────────────

class _GATWithAttention(torch.nn.Module):
    """Thin wrapper that returns attention weights from GAT layer 3."""
    def __init__(self, base):
        super().__init__()
        self.gat1 = base.gat1; self.gat2 = base.gat2; self.gat3 = base.gat3
        self.fusion = base.fusion
        self.head_tactic  = base.head_tactic
        self.head_adapt   = base.head_adapt
        self.head_suggest = base.head_suggest
        self.dr  = base.dropout_rate
        self.ctx = base.context_features

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        ctx = data.context
        bs  = batch.max().item() + 1
        if ctx.dim() == 1 and ctx.numel() == bs * self.ctx:
            ctx = ctx.view(bs, self.ctx)
        elif ctx.dim() == 3:
            ctx = ctx.squeeze(1)

        x = F.elu(self.gat1(x, ei))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = F.elu(self.gat2(x, ei))
        x = F.dropout(x, p=self.dr, training=self.training)
        x, (att_ei, att_w) = self.gat3(x, ei, return_attention_weights=True)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.fusion(torch.cat([x, ctx], 1))
        return (self.head_tactic(x), self.head_adapt(x),
                self.head_suggest(x), att_ei, att_w)


def visualise_attention(model, test_data: list, label_encoder,
                        n_examples: int = 4, save: bool = True):
    """
    Draw attention weight diagrams on a soccer pitch for one example per
    predicted tactic class. Thick amber lines = high attention edges.
    """
    att_model = _GATWithAttention(model).to(DEVICE)
    att_model.eval()

    # Find one example per predicted class
    examples = {}
    for g in test_data:
        gb = Batch.from_data_list([g]).to(DEVICE)
        with torch.no_grad():
            o1, _, _, _, _ = att_model(gb)
        pred = label_encoder.inverse_transform([o1.argmax().item()])[0]
        if pred not in examples:
            examples[pred] = g
        if len(examples) >= n_examples:
            break

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("GAT Attention Weights on Pitch — One Example per Tactic",
                 fontsize=13, fontweight="bold")

    for ax_idx, (tactic_name, g) in enumerate(examples.items()):
        ax = axes.flatten()[ax_idx]
        gb = Batch.from_data_list([g]).to(DEVICE)

        with torch.no_grad():
            o1, o2, o3, att_ei, att_w = att_model(gb)

        pred_t = label_encoder.inverse_transform([o1.argmax().item()])[0]
        pred_s = label_encoder.inverse_transform([o3.argmax().item()])[0]
        conf   = torch.softmax(o1,dim=1).max().item()*100
        adapt  = "Yes" if o2.argmax().item() == 1 else "No"

        att_m = att_w.mean(dim=1).cpu().numpy()
        att_e = att_ei.cpu().numpy()
        nx    = g.x[:,0].numpy() * PITCH_LENGTH
        ny    = g.x[:,1].numpy() * PITCH_WIDTH
        teams = g.x[:,4].numpy()

        pitch = Pitch(pitch_type="custom",
                      pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH,
                      pitch_color="#2d6a2d", line_color="white", line_alpha=0.55)
        pitch.draw(ax=ax)

        w_min = att_m.min(); w_rng = max(att_m.max()-w_min, 1e-6)
        for ei in range(att_e.shape[1]):
            src, dst = att_e[0,ei], att_e[1,ei]
            wn = (att_m[ei]-w_min)/w_rng
            if wn < 0.25:
                continue
            ax.plot([nx[src],nx[dst]], [ny[src],ny[dst]],
                    color="#EF9F27", linewidth=wn*4.5,
                    alpha=0.2+wn*0.7, solid_capstyle="round", zorder=2)

        for i in range(len(nx)):
            fc = "#1E60C8" if teams[i]==1.0 else "#C23B22"
            ec = "#7AB8FF" if teams[i]==1.0 else "#F08070"
            ax.scatter(nx[i], ny[i], s=80, color=fc, edgecolors=ec,
                       linewidths=1.5, zorder=3)

        ax.set_title(
            f"Predicted: {pred_t.replace('_',' ').title()}  ({conf:.0f}%)\n"
            f"Suggestion: {pred_s.replace('_',' ').title()}  |  Adapting: {adapt}",
            fontsize=9, color="white", pad=6, fontweight="bold"
        )
        ax.set_facecolor("#2d6a2d")

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/attention_visualisation.png",
                    dpi=130, bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR}/attention_visualisation.png")
    plt.show()


def print_results_summary(test_m: dict, df_clips, train_data,
                          val_data, test_data, n_epochs: int):
    """Print the complete results summary table to stdout."""
    print()
    print("=" * 60)
    print("TACTICAL PATTERN RECOGNITION IN SOCCER USING GAT")
    print("Rishabh Karnawat | Gisma University of Applied Sciences")
    print("=" * 60)
    print(f"\nDataset       : StatsBomb (La Liga, World Cup, Copa America)")
    print(f"Matches       : {df_clips['match_id'].nunique()}")
    print(f"Total clips   : {len(df_clips):,}")
    print(f"Train / Val / Test : {len(train_data):,} / {len(val_data):,} / {len(test_data):,}")
    print(f"Epochs        : {n_epochs}")
    print(f"\nTEST SET RESULTS")
    print("-" * 40)
    print(f"HEAD 1 Tactic     Acc: {test_m['tactic_acc']*100:.1f}%  "
          f"F1: {test_m['tactic_f1']:.3f}")
    print(f"HEAD 2 Adaptation Acc: {test_m['adapt_acc']*100:.1f}%  "
          f"F1: {test_m['adapt_f1']:.3f}")
    print(f"HEAD 3 Suggestion Acc: {test_m['suggest_acc']*100:.1f}%  "
          f"F1: {test_m['suggest_f1']:.3f}")
    print()
    print("Comparable published results (HEAD 1):")
    print("  Anzer et al. 2022 (binary only)   : ~72%")
    print("  Bauer & Anzer 2021 (binary only)  : ~68%")
    print("  TacticAI 2024 (corners only)      : ~85%")
    print(f"  This thesis (4-class, open data)  : {test_m['tactic_acc']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    print("Import and call individual functions from main.py")
