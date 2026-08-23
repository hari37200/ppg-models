"""Training and evaluation loops shared by every experiment.

Model selection uses macro-F1 rather than accuracy: with a 39/39/15/9 class
split, accuracy rewards a model for abandoning Stage 2 entirely.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hyperppg.metrics import compute_metrics

__all__ = ["TrainConfig", "fit", "evaluate", "predict_logits", "SoftTargetCrossEntropy"]


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    early_stop_patience: int = 12
    monitor: str = "macro_f1"
    verbose: bool = True
    #: Multiplier applied to the encoder/backbone LR relative to the head.
    backbone_lr_mult: float = 1.0
    history: list[dict] = field(default_factory=list)


class SoftTargetCrossEntropy(nn.Module):
    """Cross entropy against soft targets, with optional class weights.

    Needed because mixup produces fractional targets that ``nn.CrossEntropyLoss``
    cannot consume together with per-class weighting.
    """

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        if self.weight is not None:
            logp = logp * self.weight.unsqueeze(0)
            # Renormalise so the loss scale does not drift with the weights.
            denom = (target * self.weight.unsqueeze(0)).sum(-1).clamp_min(1e-8)
            return -((target * logp).sum(-1) / denom).mean()
        return -(target * logp).sum(-1).mean()


def _unpack(batch, device):
    """Split a batch into ``(inputs, tabular_or_None, targets)``."""
    if len(batch) == 3:
        x, tab, y = batch
        return x.to(device, non_blocking=True), tab.to(device, non_blocking=True), y.to(device, non_blocking=True)
    x, y = batch
    return x.to(device, non_blocking=True), None, y.to(device, non_blocking=True)


def _forward(model, x, tab):
    return model(x, tab) if tab is not None else model(x)


def _build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per batch."""
    total = max(cfg.epochs * steps_per_epoch, 1)
    warmup = max(cfg.warmup_epochs * steps_per_epoch, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_optimizer(model: nn.Module, cfg: TrainConfig):
    """AdamW, optionally with a lower LR for a pretrained backbone."""
    if cfg.backbone_lr_mult == 1.0:
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    head_names = ("head", "classifier", "fc", "tabular")
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.split(".")[0] in head_names else backbone).append(p)

    groups = [{"params": head, "lr": cfg.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult})
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; returns ``(logits, targets)``."""
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        x, tab, y = _unpack(batch, device)
        logits = _forward(model, x, tab)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.cpu().numpy())
    if not all_logits:
        return np.empty((0, 0), np.float32), np.empty((0,), np.int64)
    return np.concatenate(all_logits), np.concatenate(all_y)


def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """``(metrics, y_pred, y_true)``."""
    logits, y_true = predict_logits(model, loader, device)
    if logits.size == 0:
        return {}, np.empty(0, np.int64), y_true
    y_pred = logits.argmax(axis=1)
    return compute_metrics(y_true, y_pred), y_pred, y_true


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    cfg: TrainConfig | None = None,
    class_weight: torch.Tensor | None = None,
    num_classes: int = 4,
) -> tuple[nn.Module, dict]:
    """Train with warmup+cosine, early stopping, and best-checkpoint restore.

    Returns ``(model_with_best_weights, info)`` where ``info`` holds the best
    epoch, best metrics and the full per-epoch history.
    """
    cfg = cfg or TrainConfig()
    model = model.to(device)

    weight = class_weight.to(device) if class_weight is not None else None
    hard_criterion = nn.CrossEntropyLoss(
        weight=weight, label_smoothing=cfg.label_smoothing
    )
    soft_criterion = SoftTargetCrossEntropy(weight=weight).to(device)

    optimizer = _make_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, max(len(train_loader), 1))

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(0)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    history: list[dict] = []
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0

        for batch in train_loader:
            x, tab, y = _unpack(batch, device)
            optimizer.zero_grad(set_to_none=True)

            use_mixup = cfg.mixup_alpha > 0
            if use_mixup:
                lam = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
                if tab is not None:
                    tab = lam * tab + (1 - lam) * tab[perm]
                y1 = F.one_hot(y, num_classes).float()
                target = lam * y1 + (1 - lam) * y1[perm]

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _forward(model, x, tab)
                loss = soft_criterion(logits, target) if use_mixup else hard_criterion(logits, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running / max(n_seen, 1)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        score = val_metrics.get(cfg.monitor, -float("inf"))

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if cfg.verbose:
            print(
                f"  epoch {epoch:>3d} | loss {train_loss:.4f} | "
                f"val acc {val_metrics.get('accuracy', float('nan')):.4f} | "
                f"val macro-F1 {val_metrics.get('macro_f1', float('nan')):.4f}"
                f"{'  *' if improved else ''}"
            )

        if (
            cfg.early_stop_patience > 0
            and epochs_without_improvement >= cfg.early_stop_patience
        ):
            if cfg.verbose:
                print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "history": history,
    }
