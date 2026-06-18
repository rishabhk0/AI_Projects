"""
main.py — Run this file to execute the full TacticalGAT pipeline.

Steps
-----
 1. Download StatsBomb data
 2. EDA plots
 3. Generate clips (sliding windows)
 4. Apply tactic / adaptation / suggestion labels
 5. Encode labels + compute class weights
 6. Oversample rare classes + cap build_up
 7. Extract player positions
 8. Build PyG graphs + DataLoaders
 9. Train TacticalGAT (class-weighted loss)
10. Evaluate on test set — print results + save figures
11. (Optional) Architecture comparison — Exp 1
12. (Optional) Ablation study — Exp 3

All outputs save to results/.
Change any hyperparameter in src/config.py; this file stays untouched.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from config        import DEVICE, CHECKPOINT_PATH, RANDOM_SEED
from data_loader   import load_matches, load_events
from eda           import run_all_eda
from preprocessing import (
    build_clips_dataframe, apply_all_labels,
    oversample_and_cap, extract_all_positions,
    run_label_unit_tests,
)
from graph_builder import (
    split_by_match, encode_labels, compute_class_weights,
    build_pyg_dataset, make_dataloaders,
)
from model   import TacticalGAT
from train   import run_training
from evaluate import run_evaluation


def main(run_eda=True, run_arch_comparison=False, run_ablation=False):
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    # ── 1. Download ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Download StatsBomb data")
    print("="*60)
    df_matches = load_matches()
    df_events  = load_events(df_matches)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if run_eda:
        print("\n" + "="*60)
        print("STEP 2 — EDA (plots save to results/)")
        print("="*60)
        # EDA runs after clips are built (needs df_clips); deferred below

    # ── 3. Clips ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Generate sliding-window clips")
    print("="*60)
    run_label_unit_tests()
    df_clips = build_clips_dataframe(df_matches, df_events)

    # ── 4. Split by match ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Match-level train / val / test split")
    print("="*60)
    train_ids, val_ids, test_ids = split_by_match(df_clips)

    # ── 5. Labels ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Apply tactic / adaptation / suggestion labels")
    print("="*60)
    df_clips = apply_all_labels(df_clips, train_ids)

    # Run EDA now that df_clips exists
    if run_eda:
        run_all_eda(df_events, df_clips)

    # ── 6. Encode + class weights ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Encode labels + compute class weights")
    print("="*60)
    df_clips, le       = encode_labels(df_clips)
    class_weights      = compute_class_weights(df_clips, train_ids, le)

    # ── 7. Oversample + cap ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7 — Oversample rare classes + cap build_up")
    print("="*60)
    df_clips = oversample_and_cap(df_clips, train_ids, le)

    # ── 8. Player positions ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8 — Extract player positions from events")
    print("="*60)
    df_clips = extract_all_positions(df_clips, df_events)

    # ── 9. PyG graphs + DataLoaders ───────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 9 — Build PyG graphs + DataLoaders")
    print("="*60)
    pyg_dataset = build_pyg_dataset(df_clips, train_ids)
    (train_loader, val_loader, test_loader,
     train_data, val_data, test_data) = make_dataloaders(
        pyg_dataset, train_ids, val_ids
    )

    # ── 10. Train ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 10 — Train TacticalGAT")
    print("="*60)
    model   = TacticalGAT().to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")
    history = run_training(model, train_loader, val_loader, class_weights)

    # ── 11. Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 11 — Evaluate on test set")
    print("="*60)
    test_m = run_evaluation(
        model, test_loader, test_data, le, class_weights, history
    )

    # ── 12. Optional experiments ──────────────────────────────────────────────
    if run_arch_comparison:
        print("\n" + "="*60)
        print("STEP 12a — Architecture comparison (Experiment 1)")
        print("="*60)
        from ablation import run_architecture_comparison
        run_architecture_comparison(train_loader, val_loader, test_loader)

    if run_ablation:
        print("\n" + "="*60)
        print("STEP 12b — Ablation study (Experiment 3)")
        print("="*60)
        from ablation import run_ablation_study, print_published_work_comparison
        print_published_work_comparison()
        run_ablation_study(train_loader, val_loader, test_loader, class_weights)

    print("\n" + "="*60)
    print("DONE — all outputs saved to results/")
    print("="*60)
    return test_m


if __name__ == "__main__":
    # Set run_arch_comparison=True or run_ablation=True to run those experiments
    main(
        run_eda=True,
        run_arch_comparison=False,
        run_ablation=False,
    )
