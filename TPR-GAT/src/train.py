"""
train.py
Training and evaluation loops for TacticalGAT and all baselines.

Run standalone:
    python src/train.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    DEVICE, N_EPOCHS, LR, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR,
    GRAD_CLIP, W_TACTIC, W_ADAPT, W_SUGGEST, CHECKPOINT_PATH, RESULTS_DIR
)


def _squeeze_targets(batch):
    """Handle both batched and single-item squeeze edge cases."""
    y_t = batch.y_tactic.squeeze()
    y_a = batch.y_adapt.squeeze()
    y_s = batch.y_suggest.squeeze()
    if y_t.dim() == 0: y_t = y_t.unsqueeze(0)
    if y_a.dim() == 0: y_a = y_a.unsqueeze(0)
    if y_s.dim() == 0: y_s = y_s.unsqueeze(0)
    return y_t, y_a, y_s


def compute_loss(out_t, out_a, out_s, y_t, y_a, y_s):
    """Weighted multi-task cross-entropy loss."""
    return (W_TACTIC  * F.cross_entropy(out_t, y_t) +
            W_ADAPT   * F.cross_entropy(out_a, y_a) +
            W_SUGGEST * F.cross_entropy(out_s, y_s))


def train_one_epoch(model, loader, optimizer, device):
    """One training pass over the loader. Returns average loss."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out_t, out_a, out_s = model(batch)
        y_t, y_a, y_s       = _squeeze_targets(batch)
        loss = compute_loss(out_t, out_a, out_s, y_t, y_a, y_s)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += loss.item()
        n     += 1
    return total / max(n, 1)


def evaluate_model(model, loader, device):
    """
    Evaluate on val or test set.
    Returns dict with loss, accuracy, and macro F1 for all three heads,
    plus raw preds/trues for confusion matrices.
    """
    model.eval()
    total, n = 0.0, 0
    preds = {"t": [], "a": [], "s": []}
    trues = {"t": [], "a": [], "s": []}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out_t, out_a, out_s = model(batch)
            y_t, y_a, y_s       = _squeeze_targets(batch)

            total += compute_loss(out_t, out_a, out_s, y_t, y_a, y_s).item()
            n     += 1

            preds["t"].extend(out_t.argmax(1).cpu().numpy())
            preds["a"].extend(out_a.argmax(1).cpu().numpy())
            preds["s"].extend(out_s.argmax(1).cpu().numpy())
            trues["t"].extend(y_t.cpu().numpy())
            trues["a"].extend(y_a.cpu().numpy())
            trues["s"].extend(y_s.cpu().numpy())

    return {
        "loss":        total / max(n, 1),
        "tactic_acc":  accuracy_score(trues["t"], preds["t"]),
        "adapt_acc":   accuracy_score(trues["a"], preds["a"]),
        "suggest_acc": accuracy_score(trues["s"], preds["s"]),
        "tactic_f1":   f1_score(trues["t"], preds["t"], average="macro", zero_division=0),
        "adapt_f1":    f1_score(trues["a"], preds["a"], average="macro", zero_division=0),
        "suggest_f1":  f1_score(trues["s"], preds["s"], average="macro", zero_division=0),
        "preds": preds,
        "trues": trues,
    }


def run_training(model, train_loader, val_loader, test_loader,
                 n_epochs=N_EPOCHS, checkpoint_path=CHECKPOINT_PATH,
                 verbose=True):
    """
    Full training loop with checkpointing and scheduler.
    Returns (test_metrics, history_list).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val  = float("inf")
    history   = []

    if verbose:
        print(f"Training on {DEVICE} for {n_epochs} epochs")
        print(f"{'Ep':>4} {'Train':>10} {'Val':>9} "
              f"{'Tac':>8} {'Ada':>8} {'Sug':>8} {'F1':>7}")
        print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        vm      = evaluate_model(model, val_loader, DEVICE)
        scheduler.step(vm["loss"])

        if vm["loss"] < best_val:
            best_val = vm["loss"]
            torch.save(model.state_dict(), checkpoint_path)

        rec = {"epoch": epoch, "train_loss": tr_loss,
               **{k: v for k, v in vm.items() if k not in ("preds","trues")}}
        history.append(rec)

        if verbose:
            print(f"{epoch:>4} {tr_loss:>10.4f} {vm['loss']:>9.4f} "
                  f"{vm['tactic_acc']:>8.4f} {vm['adapt_acc']:>8.4f} "
                  f"{vm['suggest_acc']:>8.4f} {vm['tactic_f1']:>7.4f}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_m = evaluate_model(model, test_loader, DEVICE)

    if verbose:
        print(f"\nBest val loss : {best_val:.4f}")
        print(f"HEAD 1 Tactic : acc={test_m['tactic_acc']*100:.1f}%  "
              f"F1={test_m['tactic_f1']:.3f}")
        print(f"HEAD 2 Adapt  : acc={test_m['adapt_acc']*100:.1f}%   "
              f"F1={test_m['adapt_f1']:.3f}")
        print(f"HEAD 3 Suggest: acc={test_m['suggest_acc']*100:.1f}%  "
              f"F1={test_m['suggest_f1']:.3f}")

    return test_m, history


def train_and_evaluate(model_class, model_name, train_loader, val_loader,
                       test_loader, n_epochs=5, model_kwargs=None):
    """
    Helper for comparison experiments. Trains a model from scratch,
    returns test metrics dict.
    """
    if model_kwargs is None:
        model_kwargs = {}
    m = model_class(**model_kwargs).to(DEVICE)
    print(f"\nTraining {model_name} ({sum(p.numel() for p in m.parameters()):,} params)")
    test_m, _ = run_training(m, train_loader, val_loader, test_loader,
                              n_epochs=n_epochs,
                              checkpoint_path=f"_tmp_{model_name.replace(' ','_')}.pt",
                              verbose=False)
    print(f"  Tac acc: {test_m['tactic_acc']*100:.1f}%  F1: {test_m['tactic_f1']:.3f}")
    return test_m


if __name__ == "__main__":
    print("Import run_training() and call it from main.py")
