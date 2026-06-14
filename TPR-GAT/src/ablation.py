"""
ablation.py
Three ablation experiments + architecture comparison tables.

Experiment 1: Architecture comparison (MLP, LSTM, GCN, GraphSAGE, GAT)
Experiment 2: Published paper comparison table
Experiment 3: Ablation study
  3a: Activation function (ELU, ReLU, LeakyReLU, Tanh)
  3b: Number of GAT layers (1, 2, 3, 4)
  3c: Edge proximity threshold (10m, 15m, 20m, 25m)

Run standalone:
    python src/ablation.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.config import DEVICE, RESULTS_DIR, BATCH_SIZE
from src.model import (
    TacticalGAT, MLPBaseline, LSTMBaseline, GCNBaseline, GraphSAGEVariant,
    TacticalGAT_CustomActivation, TacticalGAT_NLayers
)
from src.train import train_and_evaluate, evaluate_model
from src.graph_builder import build_graph, build_context_vector


def exp1_architecture_comparison(train_loader, val_loader, test_loader,
                                 gat_results: dict, n_epochs: int = 5,
                                 save: bool = True):
    """Train MLP, LSTM, GCN, GraphSAGE and compare against GAT."""
    print("\n" + "="*65)
    print("EXPERIMENT 1: ARCHITECTURE COMPARISON")
    print("="*65)

    baselines = [
        (MLPBaseline,       "MLP"),
        (LSTMBaseline,      "LSTM"),
        (GCNBaseline,       "GCN"),
        (GraphSAGEVariant,  "GraphSAGE"),
    ]
    results = {}
    for cls, name in baselines:
        results[name] = train_and_evaluate(cls, name, train_loader,
                                           val_loader, test_loader,
                                           n_epochs=n_epochs)
    results["GAT (ours)"] = gat_results

    comp = {
        "Model":       ["MLP","LSTM","GCN","GraphSAGE","GAT (ours)"],
        "Structure":   ["Flat","Sequential","Graph+Uniform","Graph+Sample","Graph+Attention"],
        "Attention":   ["No","No","No","No","Yes"],
        "Tac Acc (%)": [f"{results[m]['tactic_acc']*100:.1f}"
                        for m in ["MLP","LSTM","GCN","GraphSAGE","GAT (ours)"]],
        "Tac F1":      [f"{results[m]['tactic_f1']:.3f}"
                        for m in ["MLP","LSTM","GCN","GraphSAGE","GAT (ours)"]],
        "Adapt Acc":   [f"{results[m]['adapt_acc']*100:.1f}%"
                        for m in ["MLP","LSTM","GCN","GraphSAGE","GAT (ours)"]],
    }
    df = pd.DataFrame(comp)
    print("\nTABLE 1: Architecture Comparison")
    print(df.to_string(index=False))

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Table 1: Architecture Comparison", fontsize=13, fontweight="bold")

    models = comp["Model"]
    colors = ["#888888","#888888","#1565C0","#6A1B9A","#1F6B3A"]

    for ax, col, title in zip(axes,
                               ["Tac Acc (%)","Tac F1"],
                               ["Tactic Accuracy (%)","Macro F1 x 100"]):
        vals = [float(v.replace("%",""))*( 1 if "Acc" in col else 100)
                for v in comp[col]]
        bars = ax.bar(models, vals, color=colors, alpha=0.85,
                      edgecolor="white", width=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/comparison_table1_architecture.png",
                    dpi=130, bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR}/comparison_table1_architecture.png")
    plt.show()
    return df


def exp2_published_comparison(gat_results: dict, save: bool = True):
    """Build contextual comparison table against published works."""
    print("\n" + "="*65)
    print("EXPERIMENT 2: COMPARISON WITH PUBLISHED WORK")
    print("="*65)

    acc_str = (f"{gat_results['tactic_acc']*100:.1f}% Acc / "
               f"{gat_results['tactic_f1']:.2f} F1")
    data = {
        "Work": ["Bauer & Anzer (2021)","Anzer et al. (2022)",
                 "Rana (2021)","TacticAI (2024)",
                 "HDS-SGT (2025)","This thesis (2025)"],
        "Task": ["Binary press","Single pattern","Event detection",
                 "Corner kicks","Formation (5-class)","4-class + adapt + suggest"],
        "Data": ["Bundesliga","Bundesliga","Tracking","Prem League",
                 "Various","StatsBomb open"],
        "Best": ["~68% F1","~72% Acc","~70% F1","~85% Acc","~79% Acc", acc_str],
        "Open": ["No","No","No","No","No","Yes"],
        "Multi-task": ["No","No","No","No","No","Yes"],
    }
    df = pd.DataFrame(data)
    print("\nTABLE 2: Comparison with Published Work")
    print(df.to_string(index=False))

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(f"{RESULTS_DIR}/comparison_table2_published.csv", index=False)
        print(f"Saved: {RESULTS_DIR}/comparison_table2_published.csv")
    return df


def _quick_train(model, train_loader, val_loader, test_loader,
                 n_epochs: int = 5):
    """Mini training loop for ablation variants."""
    opt   = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best, best_state = float("inf"), None

    for _ in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            o1, o2, o3 = model(batch)
            y1 = batch.y_tactic.squeeze()
            y2 = batch.y_adapt.squeeze()
            y3 = batch.y_suggest.squeeze()
            if y1.dim()==0: y1=y1.unsqueeze(0)
            if y2.dim()==0: y2=y2.unsqueeze(0)
            if y3.dim()==0: y3=y3.unsqueeze(0)
            loss = (F.cross_entropy(o1,y1) +
                    0.5*F.cross_entropy(o2,y2) +
                    0.5*F.cross_entropy(o3,y3))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vm = evaluate_model(model, val_loader, DEVICE)
        if vm["loss"] < best:
            best = vm["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return evaluate_model(model, test_loader, DEVICE)


def exp3_ablation(train_loader, val_loader, test_loader,
                  df_clips: pd.DataFrame,
                  train_ids, val_ids, test_ids,
                  label_encoder,
                  team_history: dict,
                  n_epochs: int = 5, save: bool = True):
    """
    Full ablation study:
      3a: Activation function
      3b: Number of GAT layers
      3c: Edge proximity threshold
    """
    print("\n" + "="*65)
    print("EXPERIMENT 3: ABLATION STUDY")
    print("="*65)

    ablation_results = []

    # 3a: Activation functions
    print("\n--- 3a: Activation Function ---")
    act_map = {
        "ELU":       F.elu,
        "ReLU":      F.relu,
        "LeakyReLU": lambda x: F.leaky_relu(x, negative_slope=0.2),
        "Tanh":      torch.tanh,
    }
    for name, fn in act_map.items():
        m = TacticalGAT_CustomActivation(activation_fn=fn).to(DEVICE)
        r = _quick_train(m, train_loader, val_loader, test_loader, n_epochs)
        ablation_results.append({
            "Experiment": "Activation Function",
            "Variant":    name,
            "Default":    "Yes" if name == "ELU" else "No",
            "Tac Acc (%)": f"{r['tactic_acc']*100:.1f}",
            "Tac F1":     f"{r['tactic_f1']:.3f}",
            "Adapt Acc":  f"{r['adapt_acc']*100:.1f}%",
            "Sug Acc":    f"{r['suggest_acc']*100:.1f}%",
        })
        print(f"  {name}: {r['tactic_acc']*100:.1f}%  F1={r['tactic_f1']:.3f}")

    # 3b: Number of layers
    print("\n--- 3b: Number of GAT Layers ---")
    for n_layers in [1, 2, 3, 4]:
        m = TacticalGAT_NLayers(n_layers=n_layers).to(DEVICE)
        r = _quick_train(m, train_loader, val_loader, test_loader, n_epochs)
        ablation_results.append({
            "Experiment": "Number of GAT Layers",
            "Variant":    f"{n_layers} layers",
            "Default":    "Yes" if n_layers == 3 else "No",
            "Tac Acc (%)": f"{r['tactic_acc']*100:.1f}",
            "Tac F1":     f"{r['tactic_f1']:.3f}",
            "Adapt Acc":  f"{r['adapt_acc']*100:.1f}%",
            "Sug Acc":    f"{r['suggest_acc']*100:.1f}%",
        })
        print(f"  {n_layers} layers: {r['tactic_acc']*100:.1f}%  F1={r['tactic_f1']:.3f}")

    # 3c: Proximity threshold — requires rebuilding graphs
    print("\n--- 3c: Edge Proximity Threshold ---")
    from src.graph_builder import build_pyg_dataset, split_and_load
    for prox in [10.0, 15.0, 20.0, 25.0]:
        print(f"  Rebuilding graphs at {prox}m ...")

        # Temporarily override proximity
        import src.config as cfg
        orig = cfg.PROXIMITY_M
        cfg.PROXIMITY_M = prox

        pyg_ds = build_pyg_dataset(df_clips, team_history, label_encoder)
        trl, vl, tsl, _, _, _ = split_and_load(
            df_clips, pyg_ds, train_ids, val_ids, test_ids
        )
        cfg.PROXIMITY_M = orig

        m = TacticalGAT_NLayers(n_layers=3).to(DEVICE)
        r = _quick_train(m, trl, vl, tsl, n_epochs)
        ablation_results.append({
            "Experiment": "Edge Proximity (m)",
            "Variant":    f"{int(prox)}m",
            "Default":    "Yes" if prox == 15.0 else "No",
            "Tac Acc (%)": f"{r['tactic_acc']*100:.1f}",
            "Tac F1":     f"{r['tactic_f1']:.3f}",
            "Adapt Acc":  f"{r['adapt_acc']*100:.1f}%",
            "Sug Acc":    f"{r['suggest_acc']*100:.1f}%",
        })
        print(f"  {int(prox)}m: {r['tactic_acc']*100:.1f}%  F1={r['tactic_f1']:.3f}")

    # Print full table
    df_abl = pd.DataFrame(ablation_results)
    print("\nTABLE 3: Ablation Study")
    print("Default: ELU activation, 3 GAT layers, 15m proximity")
    print(df_abl.to_string(index=False))

    # Best per group
    for grp in df_abl["Experiment"].unique():
        sub = df_abl[df_abl["Experiment"] == grp]
        best = sub.loc[sub["Tac Acc (%)"].astype(float).idxmax()]
        print(f"  Best {grp}: {best['Variant']} ({best['Tac Acc (%)']}%)")

    # Plot
    exps = df_abl["Experiment"].unique()
    fig, axes = plt.subplots(1, len(exps), figsize=(18, 5))
    fig.suptitle("Table 3: Ablation Study", fontsize=13, fontweight="bold")
    if len(exps) == 1:
        axes = [axes]

    for ax, exp in zip(axes, exps):
        sub  = df_abl[df_abl["Experiment"] == exp]
        vars_ = sub["Variant"].tolist()
        accs  = sub["Tac Acc (%)"].astype(float).tolist()
        defs  = sub["Default"].tolist()
        cols  = ["#1F6B3A" if d == "Yes" else "#888888" for d in defs]
        bars  = ax.bar(vars_, accs, color=cols, alpha=0.85,
                       edgecolor="white", width=0.55)
        ax.set_title(exp, fontsize=11, fontweight="bold")
        ax.set_ylabel("Tactic Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        for bar, v, d in zip(bars, accs, defs):
            lbl = f"{v:.1f}%" + ("\n(default)" if d=="Yes" else "")
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    lbl, ha="center", fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    axes[-1].legend(handles=[
        Patch(facecolor="#1F6B3A", alpha=0.85, label="Default"),
        Patch(facecolor="#888888", alpha=0.85, label="Variant"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/comparison_table3_ablation.png",
                    dpi=130, bbox_inches="tight")
        print(f"Saved: {RESULTS_DIR}/comparison_table3_ablation.png")
    plt.show()
    return df_abl


if __name__ == "__main__":
    print("Import and call exp1/exp2/exp3 from main.py")
