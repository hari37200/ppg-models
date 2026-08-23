"""Cross-validation driver shared by the replication and improvement scripts.

Collects out-of-fold predictions so the final confusion matrix covers every
segment exactly once, which is far more informative on 657 samples than an
average of five small per-fold matrices.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from hyperppg.datasets import class_weights
from hyperppg.engine import TrainConfig, fit, predict_logits
from hyperppg.metrics import compute_metrics, format_fold_summary, format_report

__all__ = ["run_cv", "pick_device"]


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def run_cv(
    folds: list[tuple[np.ndarray, np.ndarray]],
    make_datasets: Callable[[np.ndarray, np.ndarray], tuple],
    make_model: Callable[[], torch.nn.Module],
    y_all: np.ndarray,
    device: torch.device,
    cfg: TrainConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    num_classes: int = 4,
    use_class_weights: bool = True,
    title: str = "run",
    out_dir: str | Path | None = None,
    save_checkpoints: bool = False,
) -> dict:
    """Train one model per fold and report pooled out-of-fold performance.

    ``make_datasets(train_idx, val_idx)`` must return ``(train_ds, val_ds)``.
    """
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    oof_pred = np.full(len(y_all), -1, dtype=np.int64)
    oof_logits: np.ndarray | None = None
    per_fold: list[dict[str, float]] = []
    fold_info: list[dict] = []
    t_start = time.time()

    pin = device.type == "cuda"

    for k, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- {title}: fold {k + 1}/{len(folds)} "
              f"({len(train_idx)} train / {len(val_idx)} val) ---")

        train_ds, val_ds = make_datasets(train_idx, val_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=len(train_ds) > batch_size,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(batch_size, 64),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
        )

        model = make_model()
        cw = (
            class_weights(y_all[train_idx], num_classes)
            if use_class_weights
            else None
        )

        model, info = fit(
            model,
            train_loader,
            val_loader,
            device,
            cfg=cfg,
            class_weight=cw,
            num_classes=num_classes,
        )

        logits, y_true = predict_logits(model, val_loader, device)
        preds = logits.argmax(axis=1)
        if oof_logits is None:
            oof_logits = np.zeros((len(y_all), logits.shape[1]), dtype=np.float32)
        oof_logits[val_idx] = logits
        oof_pred[val_idx] = preds

        metrics = compute_metrics(y_true, preds)
        per_fold.append(metrics)
        fold_info.append({"fold": k, "best_epoch": info["best_epoch"], **metrics})
        print(
            f"  fold {k + 1} -> acc {metrics['accuracy']:.4f} | "
            f"macro-F1 {metrics['macro_f1']:.4f} | "
            f"balanced acc {metrics['balanced_accuracy']:.4f}"
        )

        if save_checkpoints and out_dir:
            torch.save(
                {"state_dict": model.state_dict(), "fold": k, "metrics": metrics},
                out_dir / f"{title}_fold{k}.pt",
            )

        del model, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    covered = oof_pred >= 0
    pooled = compute_metrics(y_all[covered], oof_pred[covered])
    elapsed = time.time() - t_start

    print()
    print(format_fold_summary(per_fold, title=f"{title}: per-fold summary"))
    print()
    print(format_report(y_all[covered], oof_pred[covered], title=f"{title}: pooled out-of-fold"))
    print(f"\ntotal time: {elapsed / 60:.1f} min")

    result = {
        "title": title,
        "n_folds": len(folds),
        "per_fold": fold_info,
        "pooled": pooled,
        "elapsed_seconds": elapsed,
        "config": {
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "mixup_alpha": cfg.mixup_alpha,
            "batch_size": batch_size,
            "class_weights": use_class_weights,
        },
    }

    if out_dir:
        (out_dir / f"{title}_results.json").write_text(json.dumps(result, indent=2))
        np.save(out_dir / f"{title}_oof_pred.npy", oof_pred)
        if oof_logits is not None:
            np.save(out_dir / f"{title}_oof_logits.npy", oof_logits)
        (out_dir / f"{title}_report.txt").write_text(
            format_report(y_all[covered], oof_pred[covered], title=title)
        )
        print(f"saved to {out_dir}")

    return result
