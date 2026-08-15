"""Training: fit Equation (9) on a labelled cohort.

There is nothing to train in the deep-learning sense. The paper's model is one
straight line,

    subjective fatigue state = a + b * fatigue index          (7), (9)

fitted by least squares on sixteen participants. Training here means fitting
``a`` and ``b`` on *your* participants and writing them to a model file that
``fatigueppg.infer`` can load.

    python -m fatigueppg.extract --manifest cohort.csv -o features.csv
    python -m fatigueppg.train --features features.csv --label-col score \\
        --out models/mycohort.json

Two things this does that the paper does not, both because sixteen points is
very few: it groups by subject before fitting, so one person contributing three
recordings cannot count as three participants, and it reports an out-of-fold
correlation next to the in-sample one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALERT_THRESHOLD, BFI_ITEMS, PAPER_R_FI, PAPER_R_NHF, ZERO_FRAC
from .model import FatigueModel, paper_model
from .stats import bfi_score, linreg, pearson

__all__ = ["prepare_table", "train_model", "main"]

FEATURES = ("fatigue_index", "fatigue_index_onset", "nhf_0_10", "nhf")


def prepare_table(features, label_col=None, bfi_items=BFI_ITEMS,
                  group_col="subject", min_sqi=0.0, feature="fatigue_index"):
    """-> ``(one row per participant, name of the label column)``.

    Accepts either an explicit label column or nine BFI-Taiwan answers, from
    which the paper's "revised subjective fatigue state" is the mean of items 2
    and 3.
    """
    df = features.copy()
    if min_sqi > 0 and "sqi" in df:
        before = len(df)
        df = df[df["sqi"] >= min_sqi]
        print(f"kept {len(df)}/{before} rows above SQI {min_sqi}")

    qcols = [f"q{i}" for i in range(1, 10)]
    if label_col is None:
        if all(c in df.columns for c in qcols):
            label_col = "revised_subjective"
            df[label_col] = [bfi_score(row, items=bfi_items)
                             for row in df[qcols].to_numpy(dtype=float)]
            print(f"label: mean of BFI items {list(bfi_items)} "
                  f"({', '.join(qcols[i-1] for i in bfi_items)})")
        elif "score" in df.columns:
            label_col = "score"
        elif "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
            label_col = "label"
        else:
            raise KeyError(
                "no label found. Pass --label-col, or include q1..q9, or a "
                f"numeric 'score' column. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not in {list(df.columns)}")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col, feature])
    if not len(df):
        raise ValueError(f"no rows with both {feature} and {label_col}")

    if group_col and group_col in df.columns:
        keep = [c for c in FEATURES if c in df.columns] + [label_col]
        grouped = df.groupby(group_col, as_index=False)[keep].mean()
        print(f"grouped {len(df)} recordings into {len(grouped)} "
              f"{group_col}s before fitting")
        return grouped, label_col
    print(f"no '{group_col}' column: fitting on {len(df)} rows as independent "
          f"participants")
    return df.reset_index(drop=True), label_col


def train_model(table, label_col, feature="fatigue_index", zero_frac=ZERO_FRAC,
                name="fitted", provenance="", threshold=ALERT_THRESHOLD,
                folds=5, seed=0) -> FatigueModel:
    model = FatigueModel.fit(table[feature], table[label_col], feature=feature,
                             zero_frac=zero_frac, name=name,
                             provenance=provenance, threshold=threshold,
                             folds=folds, seed=seed)
    return model


def compare_predictors(table, label_col):
    """The paper's central claim: the index tracks fatigue, NHF does not."""
    print("\npredictor comparison (Pearson r against the subjective state):")
    print(f"  {'predictor':<22}{'r':>9}{'p':>11}{'n':>6}   paper")
    refs = {"fatigue_index": PAPER_R_FI, "nhf_0_10": PAPER_R_NHF}
    for feat in FEATURES:
        if feat not in table:
            continue
        r, p, n = pearson(table[feat], table[label_col])
        ref = refs.get(feat)
        print(f"  {feat:<22}{r:>+9.4f}{p:>11.4g}{n:>6}   "
              f"{ref if ref is not None else '-'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="fatigue-train",
        description="Fit Equation (9) -- the fatigue index to subjective "
                    "fatigue -- on your own cohort.")
    p.add_argument("--features", required=True,
                   help="CSV from fatigueppg.extract (or any table with the "
                        "feature and label columns)")
    p.add_argument("--label-col", help="column holding the subjective score. "
                                       "Omit to use q1..q9 or 'score'")
    p.add_argument("--bfi-items", default=",".join(map(str, BFI_ITEMS)),
                   help="BFI-Taiwan items to average, e.g. '2,3' (the paper's)")
    p.add_argument("--feature", default="fatigue_index", choices=list(FEATURES),
                   help="predictor to fit on")
    p.add_argument("--group", default="subject",
                   help="column to average within before fitting; '' to disable")
    p.add_argument("--min-sqi", type=float, default=0.0)
    p.add_argument("--zero-frac", type=float, default=ZERO_FRAC,
                   help="index zero point the features were extracted with; "
                        "recorded in the model so inference matches")
    p.add_argument("--threshold", type=float, default=ALERT_THRESHOLD,
                   help="rest-reminder threshold on the index")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="model name (default: file stem)")
    p.add_argument("--out", "-o", required=True, help="model JSON to write")
    p.add_argument("--plot", help="write the regression figure here")
    args = p.parse_args(argv)

    features = pd.read_csv(args.features)
    items = tuple(int(s) for s in args.bfi_items.split(",") if s.strip())
    table, label_col = prepare_table(
        features, label_col=args.label_col, bfi_items=items,
        group_col=args.group or None, min_sqi=args.min_sqi, feature=args.feature)

    name = args.name or Path(args.out).stem
    model = train_model(table, label_col, feature=args.feature,
                        zero_frac=args.zero_frac, name=name,
                        provenance=f"{Path(args.features).name}, "
                                   f"label '{label_col}', n = {len(table)}",
                        threshold=args.threshold, folds=args.folds, seed=args.seed)

    print()
    print(model.report())
    compare_predictors(table, label_col)

    paper = paper_model()
    resid = table[label_col].to_numpy(float) - paper.predict(table[args.feature])
    print(f"\nthe paper's Eq. (9) applied unchanged to this cohort: "
          f"MAE {np.nanmean(np.abs(resid)):.2f} points on the 0-10 scale")
    if model.metrics.get("n", 0) < 12:
        print(f"\nWARNING: fitted on {model.metrics['n']} participants. The "
              f"paper used 16 and that is already thin; treat these "
              f"coefficients as provisional.")

    model.save(args.out)
    print(f"\nwrote {args.out}")
    print(f"use it with:  python -m fatigueppg.infer --model {args.out} "
          f"--input <recording>")

    if args.plot:
        from .plotting import apply_style, PALETTE, INK
        plt = apply_style()
        fig, ax = plt.subplots(figsize=(6, 4.4))
        x = table[args.feature].to_numpy(float)
        y = table[label_col].to_numpy(float)
        ax.scatter(x, y, s=34, color=PALETTE[0], edgecolors="white",
                   linewidths=0.6, label="participants", zorder=3)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)
        ax.plot(xs, model.predict(xs), color=INK, linewidth=1.8,
                label=f"fit: y = {model.a:.2f} + {model.b:.2f}x", zorder=2)
        ax.plot(xs, paper.predict(xs), color=PALETTE[1], linewidth=1.6,
                linestyle="--", label="paper Eq. (9): y = 3.1 + 0.6x", zorder=2)
        ax.set_xlabel(args.feature)
        ax.set_ylabel(label_col)
        ax.set_title(f"r = {model.metrics['r']:.3f}   (paper: {PAPER_R_FI})")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
