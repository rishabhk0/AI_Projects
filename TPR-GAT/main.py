"""
main.py
Full pipeline from data loading to results.

Usage:
    python main.py                  # full run
    python main.py --skip-eda       # skip EDA plots
    python main.py --skip-ablation  # skip ablation study
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DEVICE, RANDOM_SEED, TACTIC_CLASSES, CHECKPOINT_PATH,
    N_EPOCHS, RESULTS_DIR
)
from src.data_loader   import load_all_events
from src.eda           import run_eda
from src.preprocessing import build_dataset, add_player_positions
from src.graph_builder import (
    compute_team_history, build_pyg_dataset, split_and_load
)
from src.model         import TacticalGAT
from src.train         import run_training
from src.evaluate      import (
    plot_confusion_matrices, plot_training_history,
    visualise_attention, print_results_summary
)
from src.ablation      import (
    exp1_architecture_comparison,
    exp2_published_comparison,
    exp3_ablation,
)


def main(skip_eda: bool = False, skip_ablation: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("TACTICAL PATTERN RECOGNITION IN SOCCER USING GNN")
    print("Rishabh Karnawat | Gisma University of Applied Sciences")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading StatsBomb data...")
    df_matches, df_events = load_all_events()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not skip_eda:
        print("\n[2/7] Exploratory Data Analysis...")
        run_eda(df_events, save=True)
    else:
        print("\n[2/7] EDA skipped.")

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print("\n[3/7] Preprocessing: clips + labels...")
    df_clips, le, train_ids, val_ids, test_ids = build_dataset(df_matches, df_events)

    print("\n[3/7] Extracting player positions...")
    df_clips = add_player_positions(df_clips, df_events)

    # ── 4. Graph construction ─────────────────────────────────────────────────
    print("\n[4/7] Building graphs...")
    df_train = df_clips[df_clips["match_id"].isin(train_ids)]
    team_history = compute_team_history(df_train)

    pyg_dataset = build_pyg_dataset(df_clips, team_history, le)

    (train_loader, val_loader, test_loader,
     train_data, val_data, test_data) = split_and_load(
        df_clips, pyg_dataset, train_ids, val_ids, test_ids
    )

    # ── 5. Train main model ───────────────────────────────────────────────────
    print("\n[5/7] Training TacticalGAT...")
    model = TacticalGAT().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_m, history = run_training(
        model, train_loader, val_loader, test_loader,
        n_epochs=N_EPOCHS, checkpoint_path=CHECKPOINT_PATH, verbose=True
    )

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    print("\n[6/7] Evaluation and visualisation...")
    class_names = le.classes_.tolist()

    plot_confusion_matrices(test_m, class_names, save=True)
    plot_training_history(history, save=True)
    visualise_attention(model, test_data, le, save=True)
    print_results_summary(test_m, df_clips, train_data, val_data, test_data, N_EPOCHS)

    # ── 7. Ablation study ─────────────────────────────────────────────────────
    if not skip_ablation:
        print("\n[7/7] Running ablation study...")
        exp1_architecture_comparison(
            train_loader, val_loader, test_loader,
            gat_results=test_m, n_epochs=5
        )
        exp2_published_comparison(test_m)
        exp3_ablation(
            train_loader, val_loader, test_loader,
            df_clips=df_clips,
            train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
            label_encoder=le, team_history=team_history, n_epochs=5
        )
    else:
        print("\n[7/7] Ablation skipped.")

    print("\nAll done. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    skip_eda      = "--skip-eda"      in sys.argv
    skip_ablation = "--skip-ablation" in sys.argv
    main(skip_eda=skip_eda, skip_ablation=skip_ablation)
