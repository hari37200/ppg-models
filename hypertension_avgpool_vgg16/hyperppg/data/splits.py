"""Cross-validation splits, with and without subject leakage.

Two schemes are supported and the difference between them is the single most
important number in this project:

``subject``  StratifiedGroupKFold grouped on ``subject_id``. A subject's three
             segments always land on the same side. This is the honest protocol
             and the one every reported "improved" number uses.

``segment``  Plain StratifiedKFold over the 657 segments. Segments 1, 2 and 3
             of the same subject -- recorded seconds apart, same person, same
             label -- get scattered across train and test. This reproduces the
             paper's setup and inflates accuracy substantially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

__all__ = ["make_folds", "make_holdout", "describe_folds", "leakage_report"]

Fold = tuple[np.ndarray, np.ndarray]


def make_folds(
    index: pd.DataFrame,
    scheme: str = "subject",
    n_splits: int = 5,
    seed: int = 0,
) -> list[Fold]:
    """Return ``n_splits`` ``(train_idx, val_idx)`` positional-index pairs."""
    y = index["y"].to_numpy()
    groups = index["subject_id"].to_numpy()

    if scheme == "subject":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        return [
            (tr.astype(np.int64), va.astype(np.int64))
            for tr, va in splitter.split(np.zeros(len(y)), y, groups=groups)
        ]

    if scheme == "segment":
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        return [
            (tr.astype(np.int64), va.astype(np.int64))
            for tr, va in splitter.split(np.zeros(len(y)), y)
        ]

    raise ValueError(f"unknown scheme {scheme!r}; use 'subject' or 'segment'")


def make_holdout(
    index: pd.DataFrame,
    scheme: str = "subject",
    test_size: float = 0.2,
    seed: int = 0,
) -> Fold:
    """A single train/test split under the same two schemes."""
    y = index["y"].to_numpy()
    groups = index["subject_id"].to_numpy()
    pos = np.arange(len(index))

    if scheme == "subject":
        # GroupShuffleSplit cannot stratify; stratify on subject-level labels
        # by splitting the unique subjects instead.
        per_subject = index.drop_duplicates("subject_id")
        sub_ids = per_subject["subject_id"].to_numpy()
        sub_y = per_subject["y"].to_numpy()
        try:
            tr_sub, te_sub = train_test_split(
                sub_ids, test_size=test_size, random_state=seed, stratify=sub_y
            )
        except ValueError:
            # Falls back when a class is too small to stratify.
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed
            )
            tr, te = next(gss.split(pos, y, groups=groups))
            return tr.astype(np.int64), te.astype(np.int64)
        te_mask = np.isin(groups, te_sub)
        return pos[~te_mask].astype(np.int64), pos[te_mask].astype(np.int64)

    if scheme == "segment":
        tr, te = train_test_split(
            pos, test_size=test_size, random_state=seed, stratify=y
        )
        return tr.astype(np.int64), te.astype(np.int64)

    raise ValueError(f"unknown scheme {scheme!r}; use 'subject' or 'segment'")


def leakage_report(index: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    """Count subjects appearing on both sides of a split."""
    tr_sub = set(index.iloc[train_idx]["subject_id"].tolist())
    va_sub = set(index.iloc[val_idx]["subject_id"].tolist())
    shared = tr_sub & va_sub
    return {
        "n_train_segments": int(len(train_idx)),
        "n_val_segments": int(len(val_idx)),
        "n_train_subjects": len(tr_sub),
        "n_val_subjects": len(va_sub),
        "n_shared_subjects": len(shared),
        "leaked": len(shared) > 0,
    }


def describe_folds(index: pd.DataFrame, folds: list[Fold]) -> str:
    """Readable per-fold summary including the leakage check."""
    lines = [
        f"{'fold':>4} {'train':>7} {'val':>6} {'trn subj':>9} "
        f"{'val subj':>9} {'shared':>7}"
    ]
    for i, (tr, va) in enumerate(folds):
        r = leakage_report(index, tr, va)
        lines.append(
            f"{i:>4} {r['n_train_segments']:>7} {r['n_val_segments']:>6} "
            f"{r['n_train_subjects']:>9} {r['n_val_subjects']:>9} "
            f"{r['n_shared_subjects']:>7}"
        )
    total_shared = sum(
        leakage_report(index, tr, va)["n_shared_subjects"] for tr, va in folds
    )
    verdict = (
        "NO subject leakage"
        if total_shared == 0
        else f"SUBJECT LEAKAGE: {total_shared} shared subject-folds"
    )
    lines.append(f"-> {verdict}")
    return "\n".join(lines)
