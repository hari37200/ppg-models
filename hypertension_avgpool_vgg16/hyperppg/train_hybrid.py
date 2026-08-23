"""Train the improved 1-D CNN+transformer hybrid on PPG-BP.

Differences from the paper's route, in rough order of how much they matter:

1. Operates on the 1-D signal at 125 Hz rather than a 224x224 picture of it.
2. Subject-wise cross-validation by default, so the score is real.
3. Class-weighted loss + macro-F1 model selection, so Stage 2 is not abandoned.
4. Physiologically plausible waveform augmentation.
5. Optional encoder initialisation from self-supervised pretraining on
   PPG-DaLiA / FatigueSet (``--ssl-checkpoint``).
6. Optional fusion of clinical covariates (``--tabular``).

Examples
--------
    python -m hyperppg.train_hybrid
    python -m hyperppg.train_hybrid --ssl-checkpoint runs/ssl/ssl_encoder.pt
    python -m hyperppg.train_hybrid --ssl-checkpoint runs/ssl/ssl_encoder.pt --tabular
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hyperppg.config import default_out_dir
from hyperppg.data import ppgbp
from hyperppg.data.augment import default_train_augment
from hyperppg.data.preprocess import CLEAN_FS, clean_pipeline
from hyperppg.data.ppgbp import TABULAR_COLS
from hyperppg.data.splits import describe_folds, make_folds
from hyperppg.datasets import PPGSignalDataset
from hyperppg.engine import TrainConfig
from hyperppg.models.hybrid import PPGHybridClassifier, transfer_encoder_weights
from hyperppg.runner import pick_device, run_cv


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="subject", choices=["subject", "segment", "both"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--augment", default="strong",
                    choices=["none", "light", "medium", "strong"])
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--ssl-checkpoint", default=None,
                    help="encoder weights from hyperppg.pretrain_ssl")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=0,
                    help="reserved: currently the encoder is always trainable")
    ap.add_argument("--tabular", action="store_true",
                    help="fuse age/sex/height/weight/BMI/HR/diabetes")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    return ap


def build_tabular(index, train_idx: np.ndarray) -> np.ndarray:
    """Standardised clinical covariates, fitted on the training fold only.

    Fitting the scaler on all rows would leak test-set statistics; it is a
    small leak but a free one to avoid.
    """
    raw = index[list(TABULAR_COLS)].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    mu = raw[train_idx].mean(axis=0)
    sd = raw[train_idx].std(axis=0)
    sd[sd < 1e-8] = 1.0
    return ((raw - mu) / sd).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    out_dir = Path(args.out) if args.out else default_out_dir() / "hybrid"
    print(f"device: {device}")

    index = ppgbp.build_index(args.root)
    print(ppgbp.summarise(index))

    signals = ppgbp.load_signals(index)
    y = index["y"].to_numpy(dtype=np.int64)

    X = clean_pipeline(signals, fs_out=CLEAN_FS)
    print(f"\npreprocessed: {X.shape} "
          f"(detrend -> 0.5-8 Hz band-pass -> {CLEAN_FS:g} Hz -> z-score)")

    n_tab = len(TABULAR_COLS) if args.tabular else 0
    if args.tabular:
        print(f"tabular fusion ON: {list(TABULAR_COLS)}")
        print("  NOTE: age and BMI predict hypertension on their own, so this is")
        print("  no longer a pure-PPG result. Compare against the run without --tabular.")

    if args.ssl_checkpoint:
        ckpt = Path(args.ssl_checkpoint)
        if not ckpt.is_file():
            raise SystemExit(f"SSL checkpoint not found: {ckpt}")
        print(f"SSL init from {ckpt}")

    schemes = ["subject", "segment"] if args.split == "both" else [args.split]
    summary: dict[str, dict] = {}

    for scheme in schemes:
        folds = make_folds(index, scheme=scheme, n_splits=args.folds, seed=args.seed)
        tag = "ssl" if args.ssl_checkpoint else "scratch"
        if args.tabular:
            tag += "+tab"
        title = f"hybrid_{tag}_{scheme}"

        print(f"\n{'=' * 72}")
        print(f"{title}")
        print("=" * 72)
        print(describe_folds(index, folds))

        def make_datasets(train_idx, val_idx):
            aug = (
                None if args.augment == "none"
                else default_train_augment(fs=CLEAN_FS, strength=args.augment)
            )
            tab = build_tabular(index, train_idx) if args.tabular else None
            train_ds = PPGSignalDataset(
                X[train_idx], y[train_idx],
                tabular=tab[train_idx] if tab is not None else None,
                augment=aug,
            )
            val_ds = PPGSignalDataset(
                X[val_idx], y[val_idx],
                tabular=tab[val_idx] if tab is not None else None,
                augment=None,
            )
            return train_ds, val_ds

        def make_model():
            model = PPGHybridClassifier(num_classes=4, n_tabular=n_tab)
            if args.ssl_checkpoint:
                report = transfer_encoder_weights(
                    args.ssl_checkpoint, model, map_location="cpu", strict=True
                )
                print(f"  loaded {report['n_loaded']} encoder tensors from SSL checkpoint")
            return model

        cfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mixup_alpha=args.mixup,
            early_stop_patience=20,
            # A pretrained encoder should be nudged, not overwritten.
            backbone_lr_mult=0.3 if args.ssl_checkpoint else 1.0,
        )

        result = run_cv(
            folds=folds,
            make_datasets=make_datasets,
            make_model=make_model,
            y_all=y,
            device=device,
            cfg=cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_class_weights=not args.no_class_weights,
            title=title,
            out_dir=out_dir,
        )
        summary[title] = result["pooled"]

    print(f"\n{'=' * 72}")
    print("SUMMARY (pooled out-of-fold)")
    print("=" * 72)
    for name, m in summary.items():
        print(
            f"{name:<36} acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
            f"bal-acc {m['balanced_accuracy']:.4f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
