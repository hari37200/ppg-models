"""Self-supervised pretraining on unlabelled wrist PPG (PPG-DaLiA + FatigueSet).

Masked-span inpainting: random spans of each window are replaced by a learned
mask token and the decoder must reconstruct them. The encoder that falls out is
then fine-tuned on PPG-BP by ``train_hybrid.py --ssl-checkpoint``.

Examples
--------
    python -m hyperppg.pretrain_ssl \\
        --dalia /path/to/PPG_FieldStudy \\
        --fatigueset /path/to/fatigueset.zip \\
        --epochs 30 --out runs/ssl

Only one corpus is required; pass whichever you have.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hyperppg.config import default_out_dir
from hyperppg.data.augment import (
    AddGaussianNoise,
    AmplitudeScale,
    Compose,
    RandomApply,
    TimeShift,
)
from hyperppg.data.corpora import build_pretrain_windows, load_dalia_bvp, load_fatigueset_bvp
from hyperppg.data.preprocess import CLEAN_FS, CLEAN_SEQ_LEN
from hyperppg.datasets import PretrainWindowDataset
from hyperppg.models.hybrid import MaskedPPGAutoencoder
from hyperppg.runner import pick_device


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dalia", default=None,
                    help="PPG-DaLiA root (the directory holding S1..S15)")
    ap.add_argument("--fatigueset", default=None,
                    help="FatigueSet directory or fatigueset.zip")
    ap.add_argument("--cache", default=None,
                    help=".npy file to cache the window corpus in")
    ap.add_argument("--stride-seconds", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=120_000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--mask-span", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    return ap


def load_corpus(args) -> np.ndarray:
    """Build (or reuse) the unlabelled window corpus."""
    if args.cache:
        cache = Path(args.cache)
        if cache.is_file():
            W = np.load(cache)
            print(f"[corpus] reusing cache {cache}: {W.shape}")
            return W

    recordings: dict[str, np.ndarray] = {}
    if args.dalia:
        print("[corpus] loading PPG-DaLiA wrist BVP from E4 archives ...")
        recordings.update(
            {f"dalia_{k}": v for k, v in load_dalia_bvp(args.dalia).items()}
        )
    if args.fatigueset:
        print("[corpus] loading FatigueSet wrist BVP ...")
        recordings.update(
            {f"fatigueset_{k}": v for k, v in load_fatigueset_bvp(args.fatigueset, verbose=False).items()}
        )

    if not recordings:
        raise SystemExit(
            "no corpus specified -- pass --dalia and/or --fatigueset "
            "(see the module docstring for examples)"
        )

    total_hours = sum(v.size for v in recordings.values()) / 64.0 / 3600.0
    print(f"[corpus] {len(recordings)} recordings, {total_hours:.1f} hours of wrist PPG")

    W = build_pretrain_windows(
        recordings,
        fs_out=CLEAN_FS,
        stride_seconds=args.stride_seconds,
        max_windows=args.max_windows,
        window_len=CLEAN_SEQ_LEN,
        seed=args.seed,
        verbose=False,
    )
    print(f"[corpus] windows: {W.shape}")

    if args.cache:
        cache = Path(args.cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, W)
        print(f"[corpus] cached to {cache}")
    return W


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out) if args.out else default_out_dir() / "ssl"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    W = load_corpus(args)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(W))
    n_val = max(int(len(W) * args.val_frac), 1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[split] {len(train_idx)} train / {len(val_idx)} val windows")

    # Light augmentation only: the reconstruction target is the signal itself,
    # so anything aggressive would make the task ill-posed rather than harder.
    aug = Compose([
        RandomApply(AddGaussianNoise((20.0, 40.0)), 0.3),
        RandomApply(AmplitudeScale((0.9, 1.1)), 0.3),
        RandomApply(TimeShift(0.5), 0.5),
    ])

    train_ds = PretrainWindowDataset(W[train_idx], augment=aug)
    val_ds = PretrainWindowDataset(W[val_idx], augment=None)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = MaskedPPGAutoencoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, _, loss = model(x, mask_ratio=args.mask_ratio, span=args.mask_span)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, _, loss = model(x, mask_ratio=args.mask_ratio, span=args.mask_span)
                running += float(loss) * x.size(0)
                n += x.size(0)
        val_loss = running / max(n, 1)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    "full_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seq_len": CLEAN_SEQ_LEN,
                    "fs": CLEAN_FS,
                },
                out_dir / "ssl_encoder.pt",
            )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch:>3d} | train {train_loss:.5f} | val {val_loss:.5f} | "
            f"{time.time() - t0:.1f}s{'  *' if improved else ''}"
        )

    (out_dir / "ssl_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val loss {best_val:.5f}")
    print(f"encoder saved to {out_dir / 'ssl_encoder.pt'}")
    print("fine-tune with:")
    print(f"  python -m hyperppg.train_hybrid --ssl-checkpoint {out_dir / 'ssl_encoder.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
