"""Fit a deployable hypertension-stage classifier and write it to disk.

Everything else in this repo *evaluates*: it cross-validates and reports a
number. Nothing persists a model, so nothing can score a segment it has not
already seen. This module closes that gap, and :mod:`hyperppg.predict` consumes
what it writes.

Two fits happen here, and the distinction matters:

* **Cross-validation** (subject-wise, the honest protocol) produces the metrics
  stamped into the model card. These are what the model is worth.
* **The final fit** uses all 657 segments, because a model you are going to
  deploy should see every labelled example you have. It has no honest score of
  its own -- there is nothing left to score it on. It inherits the CV number.

Usage
-----
    python -m hyperppg.fit_model
    python -m hyperppg.fit_model --model lightgbm --out models/lgbm.joblib
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hyperppg.config import CLASS_NAMES, FS_RAW, N_SAMPLES_RAW
from hyperppg.data import ppgbp
from hyperppg.data.features import extract_features_batch
from hyperppg.data.preprocess import bandpass, detrend
from hyperppg.data.splits import describe_folds, make_folds
from hyperppg.features_baseline import make_classifier
from hyperppg.metrics import compute_metrics, format_report
from hyperppg.predict import signal_hash

DEFAULT_OUT = Path("models") / "hypertension_hgb.joblib"

#: Bumped whenever the saved layout changes in a way predict.py must notice.
SCHEMA = 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=None, help="PPG-BP dataset root")
    ap.add_argument("--model", default="hgb",
                    choices=["hgb", "lightgbm", "rf", "logreg"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-cv", action="store_true",
                    help="fit only; leaves the model card without honest metrics")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    index = ppgbp.build_index(args.root)
    print(ppgbp.summarise(index))

    signals = ppgbp.load_signals(index)
    y = index["y"].to_numpy(dtype=np.int64)

    print("\nextracting morphology features ...")
    conditioned = bandpass(detrend(signals), fs=FS_RAW, low=0.5, high=8.0)
    F, names = extract_features_batch(conditioned, fs=FS_RAW, progress=True)
    print(f"features: {F.shape} ({len(names)} columns)")

    metrics: dict[str, float] = {}
    oof_table: dict | None = None
    if not args.skip_cv:
        # Subject-wise, so the number on the model card is one the model can
        # actually deliver on someone it has never met.
        folds = make_folds(index, scheme="subject", n_splits=args.folds,
                           seed=args.seed)
        print(f"\n{'=' * 72}\nsubject-wise cross-validation\n{'=' * 72}")
        print(describe_folds(index, folds))

        oof = np.full(len(y), -1, dtype=np.int64)
        # Keep the held-out probabilities, not just the labels. They are the
        # only honest score any training segment can ever be given, and
        # predict.py serves them back for --from-dataset instead of asking the
        # final model to grade its own homework.
        oof_proba = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float32)
        for k, (tr, va) in enumerate(folds):
            clf = make_classifier(args.model, args.seed, "balanced")
            clf.fit(F[tr], y[tr])
            p = clf.predict_proba(F[va])
            for j, cls in enumerate(clf.classes_):
                oof_proba[va, int(cls)] = p[:, j]
            oof[va] = oof_proba[va].argmax(axis=1)
            m = compute_metrics(y[va], oof[va])
            print(f"  fold {k + 1}: acc {m['accuracy']:.4f} | "
                  f"macro-F1 {m['macro_f1']:.4f}")

        print()
        print(format_report(y, oof, title="pooled out-of-fold"))
        metrics = compute_metrics(y, oof)
        # Fingerprint every training waveform. predict.py hashes whatever it is
        # handed and checks this table, so a training segment gets its honest
        # held-out score no matter which way it arrives -- as a file, or by
        # subject id. Without this, handing the model its own training data back
        # returns a 99%-confident memory and reads like a triumph.
        oof_table = {
            "subject_id": index["subject_id"].to_numpy(dtype=np.int64),
            "segment": index["segment"].to_numpy(dtype=np.int64),
            "proba": oof_proba,
            "y": y,
            "hashes": [signal_hash(sig) for sig in signals],
        }

    print(f"\nfitting the final model on all {len(y)} segments ...")
    clf = make_classifier(args.model, args.seed, "balanced")
    clf.fit(F, y)

    bundle = {
        "schema": SCHEMA,
        "classifier": clf,
        "feature_names": list(names),
        "class_names": list(CLASS_NAMES),
        "fs": float(FS_RAW),
        "n_samples": int(N_SAMPLES_RAW),
        "preprocess": "detrend -> butterworth bandpass 0.5-8 Hz (zero-phase)",
        "model": args.model,
        "n_train_segments": int(len(y)),
        "n_train_subjects": int(index["subject_id"].nunique()),
        "cv": {"scheme": "subject", "folds": args.folds, **metrics},
        "oof": oof_table,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": (
            "PPG-BP (figshare 5459299), 657 fingertip segments from 219 "
            "subjects. Final fit uses every segment; the metrics above are "
            "subject-wise out-of-fold and are the honest estimate."
        ),
    }

    import joblib

    joblib.dump(bundle, out, compress=3)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    if metrics:
        print(f"  honest (subject-wise) accuracy {metrics['accuracy']:.4f} | "
              f"macro-F1 {metrics['macro_f1']:.4f}")
    print(json.dumps({k: v for k, v in bundle.items()
                      if k not in ("classifier", "feature_names", "oof")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
