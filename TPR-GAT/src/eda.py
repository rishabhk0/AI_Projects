"""
eda.py
Exploratory Data Analysis — four plots that validate the data
before building the model.

Run standalone:
    python src/eda.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from src.config import PITCH_LENGTH, PITCH_WIDTH, HALFWAY_X, RESULTS_DIR

TACTICAL_EVENTS = ["Pressure", "Carry", "Pass", "Ball Recovery"]


def plot_event_distribution(df_events: pd.DataFrame, save: bool = True):
    """Top-20 event types + four tactical types side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("StatsBomb Event Distribution", fontsize=14, fontweight="bold")

    tc = df_events["type"].value_counts().head(20)
    axes[0].barh(tc.index[::-1], tc.values[::-1],
                 color="#1F6B3A", alpha=0.85, edgecolor="white")
    axes[0].set_title("Top 20 Event Types", fontsize=12)
    axes[0].set_xlabel("Count")
    axes[0].grid(axis="x", alpha=0.3)

    tac = (df_events[df_events["type"].isin(TACTICAL_EVENTS)]
           ["type"].value_counts())
    axes[1].bar(tac.index, tac.values,
                color=["#E65100", "#1565C0", "#1F6B3A", "#6A1B9A"],
                alpha=0.85, edgecolor="white")
    axes[1].set_title("Four Tactical Event Types", fontsize=12)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    for i, (k, v) in enumerate(tac.items()):
        axes[1].text(i, v + 200, f"{v:,}", ha="center",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/eda_event_distribution.png",
                    dpi=130, bbox_inches="tight")
    plt.show()


def plot_pressure_heatmap(df_events: pd.DataFrame, save: bool = True):
    """
    Heatmap of pressure event locations on the pitch.
    Validates the High Press rule: pressures cluster in the opponent half.
    """
    pr = df_events[df_events["type"] == "Pressure"].copy()
    pr = pr[pr["location"].notna()]
    pr["x"] = pr["location"].apply(lambda l: l[0] if isinstance(l, list) else None)
    pr["y"] = pr["location"].apply(lambda l: l[1] if isinstance(l, list) else None)
    pr = pr.dropna(subset=["x", "y"])

    fig, ax = plt.subplots(figsize=(12, 7))
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#2d6a2d",
                  line_color="white", line_alpha=0.6)
    pitch.draw(ax=ax)
    bs = pitch.bin_statistic(pr["x"], pr["y"], statistic="count", bins=(30, 20))
    pitch.heatmap(bs, ax=ax, cmap="YlOrRd", alpha=0.75)

    pct = (pr["x"] > HALFWAY_X).mean() * 100
    ax.text(3, 4, f"Opp-half pressures: {pct:.1f}%",
            color="white", fontsize=10, fontweight="bold")
    ax.set_title("Pressure Events — Spatial Distribution",
                 fontsize=13, fontweight="bold", color="white")
    ax.set_facecolor("#2d6a2d")
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/eda_pressure_heatmap.png",
                    dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Pressures in opponent half: {pct:.1f}%")


def compute_carry_distances(df_events: pd.DataFrame) -> pd.DataFrame:
    """Returns carries DataFrame with carry_distance column."""
    carries = df_events[df_events["type"] == "Carry"].copy()
    carries = carries[carries["carry_end_location"].notna()]

    def dist(row):
        try:
            e = row["carry_end_location"]
            s = row["location"]
            if e and s:
                return np.sqrt((e[0]-s[0])**2 + (e[1]-s[1])**2)
        except Exception:
            pass
        return None

    carries["carry_distance"] = carries.apply(dist, axis=1)
    carries = carries.dropna(subset=["carry_distance"])
    return carries


def plot_carry_distribution(carries: pd.DataFrame, save: bool = True):
    """Histogram of carry distances. Marks the 20m progressive carry threshold."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(carries["carry_distance"], bins=50, color="#1565C0",
            alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(x=20, color="#E65100", linewidth=2.5, linestyle="--",
               label="Progressive carry threshold (20m)")
    ax.set_xlabel("Carry distance (metres)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Carry Distance Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    pct = (carries["carry_distance"] > 20).mean() * 100
    ax.text(25, ax.get_ylim()[1] * 0.85,
            f"{pct:.1f}% are progressive\n(>20m)",
            color="#E65100", fontsize=10, fontweight="bold")
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/eda_carry_distribution.png",
                    dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Progressive carries (>20m): {pct:.1f}%  |  Total: {len(carries):,}")


def plot_team_tendencies(df_events: pd.DataFrame,
                         carries: pd.DataFrame, save: bool = True):
    """
    Per-team rates for each of the four tactical signals.
    Validates that event patterns align with known team styles.
    """
    stats = []
    for team in df_events["team"].unique():
        te    = df_events[df_events["team"] == team]
        total = max(len(te), 1)

        locs_p    = te[te["type"] == "Pressure"]["location"].dropna()
        opp_press = sum(1 for l in locs_p if isinstance(l, list) and l[0] > HALFWAY_X)

        car_te   = carries[carries["team"] == team]
        prog_car = (car_te["carry_distance"] > 20).sum()

        locs_pa  = te[te["type"] == "Pass"]["location"].dropna()
        own_pass = sum(1 for l in locs_pa if isinstance(l, list) and l[0] < HALFWAY_X)

        rec = len(te[te["type"] == "Ball Recovery"])
        stats.append({
            "team":       team,
            "press":      round(opp_press / total * 100, 2),
            "prog_carry": round(prog_car   / total * 100, 2),
            "own_pass":   round(own_pass   / total * 100, 2),
            "recovery":   round(rec        / total * 100, 2),
        })

    df = pd.DataFrame(stats)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Team Tactical Indicators", fontsize=14, fontweight="bold")

    for ax, col, title, color in zip(
        axes.flatten(),
        ["press", "prog_carry", "own_pass", "recovery"],
        ["High Press signal", "Counter-Attack signal",
         "Low Block signal",  "Transition signal"],
        ["#E65100", "#1565C0", "#6A1B9A", "#1F6B3A"]
    ):
        data = df.set_index("team")[col].sort_values()
        ax.barh(data.index, data.values, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/eda_team_tendencies.png",
                    dpi=130, bbox_inches="tight")
    plt.show()


def run_eda(df_events: pd.DataFrame, save: bool = True):
    """Run all four EDA plots. Returns carries DataFrame for downstream use."""
    print("=== 1/4 Event Distribution ===")
    plot_event_distribution(df_events, save=save)
    print("=== 2/4 Pressure Heatmap ===")
    plot_pressure_heatmap(df_events, save=save)
    print("=== 3/4 Carry Distance ===")
    carries = compute_carry_distances(df_events)
    plot_carry_distribution(carries, save=save)
    print("=== 4/4 Team Tendencies ===")
    plot_team_tendencies(df_events, carries, save=save)
    return carries


if __name__ == "__main__":
    from src.data_loader import load_all_events
    df_matches, df_events = load_all_events()
    run_eda(df_events)
