"""Replicate the paper: AlexNet / ResNet-50 / VGG-16 / AvgPool_VGG-16.

Reproduces the paper's pipeline -- moving-average smoothing, min-max
normalisation, waveform rendered as an image, ImageNet-pretrained CNN -- under
both evaluation protocols.

Examples
--------
    # the paper's own protocol (segment-level split -> subject leakage)
    python -m hyperppg.train_paper --model avgpool_vgg16 --split segment

    # the honest protocol
    python -m hyperppg.train_paper --model avgpool_vgg16 --split subject

    # every model, both protocols
    python -m hyperppg.train_paper --model all --split both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hyperppg.config import default_out_dir
from hyperppg.data import ppgbp
from hyperppg.data.augment import default_train_augment
from hyperppg.data.preprocess import paper_pipeline
from hyperppg.data.splits import describe_folds, make_folds
from hyperppg.datasets import PPGImageDataset
from hyperppg.engine import TrainConfig
from hyperppg.models.paper import PAPER_MODELS, build_model, count_parameters
from hyperppg.runner import pick_device, run_cv


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="PPG-BP dataset root (auto-detected)")
    ap.add_argument("--model", default="avgpool_vgg16",
                    choices=[*PAPER_MODELS, "all"])
    ap.add_argument("--split", default="subject",
                    choices=["subject", "segment", "both"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--augment", default="medium",
                    choices=["none", "light", "medium", "strong"])
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="train from scratch instead of ImageNet weights")
    ap.add_argument("--freeze-features", action="store_true",
                    help="train only the classifier head")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    out_dir = Path(args.out) if args.out else default_out_dir() / "paper"
    print(f"device: {device}")

    index = ppgbp.build_index(args.root)
    print(ppgbp.summarise(index))

    signals = ppgbp.load_signals(index)
    y = index["y"].to_numpy(dtype=np.int64)

    # The paper's preprocessing, verbatim.
    X = paper_pipeline(signals, window=50)
    print(f"\npreprocessed: {X.shape} (moving average w=50, min-max to [0, 1])")

    models = list(PAPER_MODELS) if args.model == "all" else [args.model]
    schemes = ["subject", "segment"] if args.split == "both" else [args.split]

    summary: dict[str, dict] = {}

    for scheme in schemes:
        folds = make_folds(index, scheme=scheme, n_splits=args.folds, seed=args.seed)
        print(f"\n{'=' * 72}")
        print(f"SPLIT SCHEME: {scheme}")
        print("=" * 72)
        print(describe_folds(index, folds))

        for model_name in models:
            title = f"{model_name}_{scheme}"
            print(f"\n{'#' * 72}")
            print(f"# {title}")
            print("#" * 72)

            probe = build_model(
                model_name,
                num_classes=4,
                pretrained=not args.no_pretrained,
                freeze_features=args.freeze_features,
            )
            total, trainable = count_parameters(probe)
            print(f"parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
            del probe

            def make_datasets(train_idx, val_idx):
                aug = (
                    None if args.augment == "none"
                    else default_train_augment(fs=1000.0, strength=args.augment)
                )
                train_ds = PPGImageDataset(
                    X[train_idx], y[train_idx], augment=aug,
                    height=args.img_size, width=args.img_size,
                    line_width=args.line_width, seed_offset=0,
                )
                val_ds = PPGImageDataset(
                    X[val_idx], y[val_idx], augment=None,
                    height=args.img_size, width=args.img_size,
                    line_width=args.line_width,
                )
                return train_ds, val_ds

            def make_model(name=model_name):
                return build_model(
                    name,
                    num_classes=4,
                    pretrained=not args.no_pretrained,
                    freeze_features=args.freeze_features,
                )

            cfg = TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                mixup_alpha=args.mixup,
                # A pretrained ImageNet stem should move slower than a fresh head.
                backbone_lr_mult=0.1 if not args.no_pretrained else 1.0,
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
    print(f"{'run':<32} {'accuracy':>9} {'macro-F1':>9} {'bal-acc':>9}")
    for name, m in summary.items():
        print(
            f"{name:<32} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
            f"{m['balanced_accuracy']:>9.4f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
