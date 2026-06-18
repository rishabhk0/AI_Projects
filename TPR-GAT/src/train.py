"""
train.py — Training loop with class-weighted loss.

Public API
----------
train_one_epoch(model, loader, optimizer, device, class_weights_tensor) -> float
evaluate_model(model, loader, device, class_weights_tensor)              -> dict
run_training(model, train_loader, val_loader, class_weights_tensor)      -> list[dict]
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from config import DEVICE, N_EPOCHS, LR, WEIGHT_DECAY, CHECKPOINT_PATH
import os


def train_one_epoch(model, loader, optimizer, device, class_weights_tensor):
    """
    One training epoch.
    HEAD 1 uses weighted CE so counter_attack / high_press are penalised heavily.
    HEAD 2 and HEAD 3 use standard CE.
    """
    model.train()
    total, nb = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        o1, o2, o3 = model(batch)
        y1 = batch.y_tactic.squeeze()
        y2 = batch.y_adapt.squeeze()
        y3 = batch.y_suggest.squeeze()
        if y1.dim() == 0: y1 = y1.unsqueeze(0)
        if y2.dim() == 0: y2 = y2.unsqueeze(0)
        if y3.dim() == 0: y3 = y3.unsqueeze(0)

        loss = (
            1.0 * F.cross_entropy(o1, y1, weight=class_weights_tensor) +
            0.5 * F.cross_entropy(o2, y2) +
            0.5 * F.cross_entropy(o3, y3)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item(); nb += 1

    return total / max(nb, 1)


def evaluate_model(model, loader, device, class_weights_tensor):
    """
    Evaluate on a DataLoader. Returns metrics dict with per-class F1
    for HEAD 1 so you can track rare-class learning each epoch.
    """
    model.eval()
    total, nb = 0.0, 0
    preds = {"t": [], "a": [], "s": []}
    trues = {"t": [], "a": [], "s": []}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            o1, o2, o3 = model(batch)
            y1 = batch.y_tactic.squeeze()
            y2 = batch.y_adapt.squeeze()
            y3 = batch.y_suggest.squeeze()
            if y1.dim() == 0: y1 = y1.unsqueeze(0)
            if y2.dim() == 0: y2 = y2.unsqueeze(0)
            if y3.dim() == 0: y3 = y3.unsqueeze(0)

            loss = (
                1.0 * F.cross_entropy(o1, y1, weight=class_weights_tensor) +
                0.5 * F.cross_entropy(o2, y2) +
                0.5 * F.cross_entropy(o3, y3)
            )
            total += loss.item(); nb += 1

            preds["t"].extend(o1.argmax(1).cpu().numpy())
            preds["a"].extend(o2.argmax(1).cpu().numpy())
            preds["s"].extend(o3.argmax(1).cpu().numpy())
            trues["t"].extend(y1.cpu().numpy())
            trues["a"].extend(y2.cpu().numpy())
            trues["s"].extend(y3.cpu().numpy())

    per_cls_f1 = f1_score(trues["t"], preds["t"],
                          labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "loss":              total / max(nb, 1),
        "tactic_acc":        accuracy_score(trues["t"], preds["t"]),
        "adapt_acc":         accuracy_score(trues["a"], preds["a"]),
        "suggest_acc":       accuracy_score(trues["s"], preds["s"]),
        "tactic_f1":         f1_score(trues["t"], preds["t"], average="macro", zero_division=0),
        "adapt_f1":          f1_score(trues["a"], preds["a"], average="macro", zero_division=0),
        "suggest_f1":        f1_score(trues["s"], preds["s"], average="macro", zero_division=0),
        "f1_build_up":       float(per_cls_f1[0]),
        "f1_counter_attack": float(per_cls_f1[1]),
        "f1_high_press":     float(per_cls_f1[2]),
        "f1_low_block":      float(per_cls_f1[3]),
        "preds": preds,
        "trues": trues,
    }


def run_training(model, train_loader, val_loader,
                 class_weights_tensor,
                 n_epochs=N_EPOCHS,
                 checkpoint_path=CHECKPOINT_PATH):
    """Train for n_epochs, save best checkpoint by val loss, return history."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    history       = []

    print(f"Training TacticalGAT for {n_epochs} epochs on {DEVICE}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print()
    print(f"{'Ep':>3} {'TrLoss':>8} {'ValLoss':>8} "
          f"{'TacAcc':>8} {'MacF1':>7} "
          f"{'F1_bu':>7} {'F1_ca':>7} {'F1_hp':>7} {'F1_lb':>7}")
    print("-" * 75)

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  DEVICE, class_weights_tensor)
        vm = evaluate_model(model, val_loader, DEVICE, class_weights_tensor)
        scheduler.step(vm["loss"])

        if vm["loss"] < best_val_loss:
            best_val_loss = vm["loss"]
            torch.save(model.state_dict(), checkpoint_path)

        history.append({
            "epoch": epoch, "train_loss": tr_loss,
            **{k: v for k, v in vm.items() if k not in ("preds", "trues")},
        })

        print(f"{epoch:>3} {tr_loss:>8.4f} {vm['loss']:>8.4f} "
              f"{vm['tactic_acc']:>8.4f} {vm['tactic_f1']:>7.4f} "
              f"{vm['f1_build_up']:>7.4f} {vm['f1_counter_attack']:>7.4f} "
              f"{vm['f1_high_press']:>7.4f} {vm['f1_low_block']:>7.4f}")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    return history
