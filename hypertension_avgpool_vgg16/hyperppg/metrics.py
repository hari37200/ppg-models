"""Evaluation metrics.

Accuracy alone is misleading here -- the majority class is 39% of segments, so
a model that never predicts Stage 2 can still look respectable. Every report
therefore carries balanced accuracy and macro-F1 alongside it, and model
selection uses macro-F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from hyperppg.config import CLASS_NAMES

__all__ = ["compute_metrics", "format_report", "majority_baseline", "aggregate_folds"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Headline scalar metrics."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def majority_baseline(y: np.ndarray) -> dict[str, float]:
    """Scores of always predicting the most frequent class."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return {}
    majority = int(np.bincount(y, minlength=len(CLASS_NAMES)).argmax())
    return compute_metrics(y, np.full_like(y, majority))


def format_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> str:
    """Per-class table + confusion matrix + headline metrics, as text."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(len(class_names)))

    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]

    lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
            digits=3,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(n) for n in class_names) + 2
    lines.append("confusion matrix (rows = true, cols = predicted)")
    header = " " * width + "".join(f"{n[:9]:>10}" for n in class_names)
    lines.append(header)
    for name, row in zip(class_names, cm):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>10d}" for v in row))

    m = compute_metrics(y_true, y_pred)
    base = majority_baseline(y_true)
    lines.append("")
    lines.append(
        f"accuracy {m['accuracy']:.4f} | balanced acc {m['balanced_accuracy']:.4f} | "
        f"macro-F1 {m['macro_f1']:.4f} | weighted-F1 {m['weighted_f1']:.4f} | "
        f"kappa {m['cohen_kappa']:.4f}"
    )
    if base:
        lines.append(
            f"majority-class baseline: accuracy {base['accuracy']:.4f} | "
            f"macro-F1 {base['macro_f1']:.4f}"
        )
    return "\n".join(lines)


def aggregate_folds(per_fold: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each metric across folds."""
    if not per_fold:
        return {}
    keys = per_fold[0].keys()
    return {
        k: (float(np.mean([f[k] for f in per_fold])), float(np.std([f[k] for f in per_fold])))
        for k in keys
    }


def format_fold_summary(per_fold: list[dict[str, float]], title: str = "") -> str:
    """Readable mean +/- std table over cross-validation folds."""
    agg = aggregate_folds(per_fold)
    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"{'metric':<20} {'mean':>8} {'std':>8}")
    for k, (mu, sd) in agg.items():
        lines.append(f"{k:<20} {mu:>8.4f} {sd:>8.4f}")
    return "\n".join(lines)
