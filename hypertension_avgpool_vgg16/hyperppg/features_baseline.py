"""Gradient-boosting baseline on handcrafted PPG morphology features.

Runs in under a minute on CPU and is the number every deep model in this repo
has to beat. On 657 segments with 4 imbalanced classes, a well-specified
feature set is a serious competitor -- treat a deep model that fails to clear
this bar as evidence that the deep model is not working, not that the task is
impossible.

Examples
--------
    python -m hyperppg.features_baseline
    python -m hyperppg.features_baseline --split both --tabular
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hyperppg.config import FS_RAW, default_out_dir
from hyperppg.data import ppgbp
from hyperppg.data.features import extract_features_batch
from hyperppg.data.ppgbp import TABULAR_COLS
from hyperppg.data.preprocess import bandpass, detrend
from hyperppg.data.splits import describe_folds, make_folds
from hyperppg.metrics import compute_metrics, format_fold_summary, format_report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="subject", choices=["subject", "segment", "both"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="hgb", choices=["hgb", "lightgbm", "rf", "logreg"])
    ap.add_argument("--tabular", action="store_true",
                    help="append age/sex/BMI/HR (no longer a pure-PPG result)")
    ap.add_argument("--class-weight", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def make_classifier(name: str, seed: int, class_weight):
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight=class_weight,
            random_state=seed,
            verbose=-1,
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight=class_weight, random_state=seed
            ),
        )
    raise ValueError(f"unknown model {name!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out) if args.out else default_out_dir() / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = ppgbp.build_index(args.root)
    print(ppgbp.summarise(index))

    signals = ppgbp.load_signals(index)
    y = index["y"].to_numpy(dtype=np.int64)

    # Condition the signal, then extract morphology at the native 1000 Hz --
    # fiducial timing resolution is what these features live on.
    print("\nextracting morphology features ...")
    conditioned = bandpass(detrend(signals), fs=FS_RAW, low=0.5, high=8.0)
    F, names = extract_features_batch(conditioned, fs=FS_RAW, progress=True)
    print(f"features: {F.shape} ({len(names)} columns)")

    if args.tabular:
        tab = index[list(TABULAR_COLS)].to_numpy(dtype=np.float64)
        tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        F = np.hstack([F, tab])
        names = names + list(TABULAR_COLS)
        print(f"with tabular covariates: {F.shape}")
        print("  NOTE: age/BMI are strong hypertension predictors on their own.")

    class_weight = "balanced" if args.class_weight else None
    schemes = ["subject", "segment"] if args.split == "both" else [args.split]
    summary: dict[str, dict] = {}

    for scheme in schemes:
        folds = make_folds(index, scheme=scheme, n_splits=args.folds, seed=args.seed)
        title = f"{args.model}_{scheme}" + ("+tab" if args.tabular else "")
        print(f"\n{'=' * 72}")
        print(title)
        print("=" * 72)
        print(describe_folds(index, folds))

        oof = np.full(len(y), -1, dtype=np.int64)
        per_fold: list[dict[str, float]] = []
        importances: list[np.ndarray] = []

        for k, (tr, va) in enumerate(folds):
            clf = make_classifier(args.model, args.seed, class_weight)
            clf.fit(F[tr], y[tr])
            pred = clf.predict(F[va])
            oof[va] = pred
            m = compute_metrics(y[va], pred)
            per_fold.append(m)
            print(f"  fold {k + 1}: acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f}")

            if hasattr(clf, "feature_importances_"):
                importances.append(np.asarray(clf.feature_importances_, dtype=np.float64))

        print()
        print(format_fold_summary(per_fold, title=f"{title}: per-fold"))
        print()
        print(format_report(y, oof, title=f"{title}: pooled out-of-fold"))

        if importances:
            mean_imp = np.mean(importances, axis=0)
            order = np.argsort(mean_imp)[::-1][:20]
            print("\ntop 20 features by mean importance")
            for rank, j in enumerate(order, 1):
                print(f"  {rank:>2}. {names[j]:<28} {mean_imp[j]:.4f}")

        summary[title] = compute_metrics(y, oof)
        np.save(out_dir / f"{title}_oof.npy", oof)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    for name, m in summary.items():
        print(
            f"{name:<28} acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
            f"bal-acc {m['balanced_accuracy']:.4f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
