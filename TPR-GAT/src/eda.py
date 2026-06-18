"""
eda.py — Exploratory data analysis. Saves 4 plots to results/.

Public API
----------
run_all_eda(df_events, df_clips)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from config import PITCH_LENGTH, PITCH_WIDTH, HALFWAY_X


def plot_event_distribution(df_events, out="results/eda_event_distribution.png"):
    counts = df_events["type"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(counts.index, counts.values,
                  color="#1F6B3A", alpha=0.85, edgecolor="white")
    ax.set_title("Event Type Distribution — All Competitions",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_pressure_heatmap(df_events, out="results/eda_pressure_heatmap.png"):
    pressures = df_events[
        (df_events["type"] == "Pressure") &
        (df_events["location"].notna())
    ]
    xs, ys = [], []
    for loc in pressures["location"]:
        if isinstance(loc, list) and len(loc) == 2:
            xs.append(float(loc[0]))
            ys.append(float(loc[1]))

    if not xs:
        print("No pressure locations found — skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    pitch = Pitch(pitch_color="grass", line_color="white",
                  pitch_type="statsbomb")
    pitch.draw(ax=ax)
    pitch.kdeplot(np.array(xs), np.array(ys), ax=ax,
                  cmap="Reds", fill=True, levels=100,
                  alpha=0.6, bw_adjust=0.7)
    ax.set_title("Pressure Event Heatmap — All Competitions",
                 fontsize=12, fontweight="bold", color="white",
                 bbox=dict(facecolor="#333333", alpha=0.7, pad=4))
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_carry_distances(df_events, out="results/eda_carry_distances.png"):
    carries   = df_events[df_events["type"] == "Carry"]
    distances = []

    for _, row in carries.iterrows():
        try:
            c  = row.get("carry")
            sl = row.get("location")
            if not (isinstance(c, dict) and isinstance(sl, list)):
                continue
            el = c.get("end_location")
            if not (el and isinstance(el, list)):
                continue
            dx = (el[0] - sl[0]) * (PITCH_LENGTH / 120.0)
            dy = (el[1] - sl[1]) * (PITCH_WIDTH  / 80.0)
            distances.append(float(np.sqrt(dx * dx + dy * dy)))
        except Exception:
            pass

    if not distances:
        print("No carry distances — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(distances, bins=60, color="#1565C0",
            alpha=0.8, edgecolor="white")
    ax.axvline(x=20, color="red", linestyle="--", linewidth=1.5,
               label="Progressive carry threshold (20 m)")
    ax.set_title("Carry Distance Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_team_tactic_tendencies(df_clips, out="results/eda_team_tendencies.png"):
    """Bar chart of tactic distribution with class imbalance clearly visible."""
    counts = df_clips["tactic_label"].value_counts()
    total  = counts.sum()
    colors = {
        "build_up":       "#1F6B3A",
        "low_block":      "#1565C0",
        "counter_attack": "#E65100",
        "high_press":     "#6A1B9A",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[colors.get(c, "#888888") for c in counts.index],
        alpha=0.85, edgecolor="white",
    )
    ax.set_title("Tactic Label Distribution — Full Dataset\n"
                 "(class imbalance is the core problem this project addresses)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Clip count")
    ax.grid(axis="y", alpha=0.3)
    for bar, (cls, val) in zip(bars, counts.items()):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def run_all_eda(df_events, df_clips):
    os.makedirs("results", exist_ok=True)
    plot_event_distribution(df_events)
    plot_pressure_heatmap(df_events)
    plot_carry_distances(df_events)
    plot_team_tactic_tendencies(df_clips)
    print("All EDA plots saved.")
