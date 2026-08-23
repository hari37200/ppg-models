"""Correlation and regression -- Sections 3.7 and 3.8.

    r = sum((Xi - Xbar)(Yi - Ybar)) / sqrt(sum(Xi - Xbar)^2 sum(Yi - Ybar)^2)   (8)
    Y = a + b * X                                                               (7)
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sst

from .config import BFI_ITEMS

__all__ = ["pearson", "linreg", "kfold_r", "bfi_score"]


def pearson(x, y):
    """Equation (8) -> ``(r, p, n)``. Non-finite pairs are dropped."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    r, p = sst.pearsonr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def linreg(x, y):
    """Equation (7) by least squares -> dict(a, b, r, p, n, stderr, rmse, mae)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return dict(a=np.nan, b=np.nan, r=np.nan, p=np.nan, n=n,
                    stderr=np.nan, rmse=np.nan, mae=np.nan)
    f = sst.linregress(x[ok], y[ok])
    resid = y[ok] - (f.intercept + f.slope * x[ok])
    return dict(a=float(f.intercept), b=float(f.slope), r=float(f.rvalue),
                p=float(f.pvalue), n=n, stderr=float(f.stderr),
                rmse=float(np.sqrt(np.mean(resid ** 2))),
                mae=float(np.mean(np.abs(resid))))


def kfold_r(x, y, k=5, seed=0):
    """Out-of-fold correlation and error for a straight-line fit.

    An in-sample r on sixteen participants is an optimistic number; this is the
    honest one. Returns ``(r_oof, mae_oof, n)``, or NaNs if there is not enough
    data to split.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < max(2 * k, 6):
        return np.nan, np.nan, n

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    pred = np.full(n, np.nan)
    for f in folds:
        train = np.setdiff1d(order, f)
        if train.size < 3:
            continue
        fit = sst.linregress(x[train], y[train])
        pred[f] = fit.intercept + fit.slope * x[f]

    ok = np.isfinite(pred)
    if ok.sum() < 3:
        return np.nan, np.nan, n
    r, _ = sst.pearsonr(pred[ok], y[ok])
    return float(r), float(np.mean(np.abs(pred[ok] - y[ok]))), int(ok.sum())


def bfi_score(answers, items=BFI_ITEMS):
    """Revised subjective fatigue state = mean of the chosen BFI-Taiwan items.

    ``answers`` holds the nine 0-10 responses in the order of the paper's
    Table 1. The paper averaged items 2 and 3, the two that correlated with the
    index (r = 0.8743 and 0.5328).
    """
    a = np.asarray(answers, dtype=float)
    if a.shape[-1] != 9:
        raise ValueError(f"expected nine BFI-Taiwan answers, got {a.shape[-1]}")
    return float(np.mean(a[..., [i - 1 for i in items]], axis=-1))
