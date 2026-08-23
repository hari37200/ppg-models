"""End-to-end sanity check of the data pipeline.

Run this first on any new machine (Colab, Kaggle, local) -- it fails loudly and
specifically rather than letting a silent shape bug reach training.

    python -m hyperppg.selfcheck
    python -m hyperppg.selfcheck --root data/ppgbp
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np

from hyperppg.config import (
    CLASS_NAMES,
    EXPECTED_SUBJECT_COUNTS,
    FS_RAW,
    N_SAMPLES_RAW,
    resolve_ppgbp_root,
)

_PASS, _FAIL = "  ok  ", " FAIL "
_results: list[tuple[str, bool, str]] = []


def _check(name: str, fn) -> object:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic harness
        _results.append((name, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc(limit=3)
        return None
    _results.append((name, True, "" if value is None else str(value)))
    return value


def run(root: str | None = None, skip_torch: bool = False) -> int:
    from hyperppg.data import ppgbp, preprocess, render, splits
    from hyperppg.data.augment import default_train_augment

    print("=" * 72)
    print("hyperppg self-check")
    print("=" * 72)

    data_root = _check("locate dataset", lambda: resolve_ppgbp_root(root))
    if data_root is None:
        _summary()
        return 1

    holder: dict = {}

    def _build():
        idx = ppgbp.build_index(data_root)
        holder["index"] = idx
        return (
            f"{len(idx)} segments x {idx.shape[1]} columns, "
            f"{idx['subject_id'].nunique()} subjects"
        )

    _check("build index", _build)
    index = holder.get("index")
    if index is None:
        _summary()
        return 1

    def _class_balance():
        per_subject = index.drop_duplicates("subject_id")
        got = {c: int((per_subject["label"] == c).sum()) for c in CLASS_NAMES}
        if got != EXPECTED_SUBJECT_COUNTS:
            raise AssertionError(f"subject counts {got} != {EXPECTED_SUBJECT_COUNTS}")
        return got

    _check("subject class balance", _class_balance)

    def _load():
        holder["X"] = ppgbp.load_signals(index)
        return f"{holder['X'].shape} {holder['X'].dtype}"

    _check("load signals", _load)
    X = holder.get("X")
    if X is None:
        _summary()
        return 1

    def _shape():
        assert X.shape == (len(index), N_SAMPLES_RAW), X.shape
        assert X.dtype == np.float32, X.dtype
        assert np.isfinite(X).all(), "non-finite samples in raw signals"
        return X.shape

    _check("raw signal shape/finiteness", _shape)

    def _paper_pipe():
        Xp = preprocess.paper_pipeline(X)
        assert Xp.shape == X.shape, Xp.shape
        assert np.isfinite(Xp).all()
        assert Xp.min() >= -1e-6 and Xp.max() <= 1 + 1e-6, (Xp.min(), Xp.max())
        return f"{Xp.shape}, range [{Xp.min():.3f}, {Xp.max():.3f}]"

    _check("paper pipeline (smooth + minmax)", _paper_pipe)

    def _clean_pipe():
        Xc = preprocess.clean_pipeline(X, fs_in=FS_RAW, fs_out=preprocess.CLEAN_FS)
        expected_len = preprocess.resampled_length(
            N_SAMPLES_RAW, FS_RAW, preprocess.CLEAN_FS
        )
        assert expected_len == preprocess.CLEAN_SEQ_LEN, expected_len
        assert Xc.shape == (len(index), expected_len), (Xc.shape, expected_len)
        assert np.isfinite(Xc).all()
        assert abs(float(Xc.mean())) < 1e-3, Xc.mean()
        return f"{Xc.shape}, mean {Xc.mean():.2e}, std {Xc.std():.3f}"

    _check("clean pipeline (bandpass + 125 Hz + zscore)", _clean_pipe)

    def _render_one():
        img = render.waveform_to_image(X[0], height=224, width=224, normalize=False)
        assert img.shape == (3, 224, 224), img.shape
        ink = float(img[0].mean())
        # A 224-wide trace should occupy a few percent of the canvas: enough to
        # prove the rasteriser drew something, little enough to prove it did not
        # flood-fill.
        assert 0.005 < ink < 0.35, f"ink fraction {ink:.4f} looks wrong"
        # Every column must contain ink -- otherwise the trace is broken.
        cols_with_ink = (img[0] > 0).any(axis=0).sum()
        assert cols_with_ink == 224, f"only {cols_with_ink}/224 columns drawn"
        return f"ink {ink:.4f}, all 224 columns drawn"

    _check("rasteriser", _render_one)

    def _render_batch():
        imgs = render.batch_to_images(X[:8], height=64, width=64)
        assert imgs.shape == (8, 3, 64, 64), imgs.shape
        assert np.isfinite(imgs).all()
        return imgs.shape

    _check("batch rasteriser", _render_batch)

    def _augment():
        rng = np.random.default_rng(0)
        aug = default_train_augment(fs=FS_RAW, strength="strong")
        out = [aug(X[0].copy(), rng) for _ in range(25)]
        for o in out:
            assert o.shape == X[0].shape, o.shape
            assert np.isfinite(o).all()
        changed = sum(not np.allclose(o, X[0]) for o in out)
        assert changed >= 20, f"only {changed}/25 augmentations altered the signal"
        return f"{changed}/25 draws altered the signal, all shapes preserved"

    _check("augmentation stack", _augment)

    def _subject_split():
        folds = splits.make_folds(index, scheme="subject", n_splits=5, seed=0)
        assert len(folds) == 5
        total_shared = 0
        covered = set()
        for tr, va in folds:
            r = splits.leakage_report(index, tr, va)
            total_shared += r["n_shared_subjects"]
            covered |= set(va.tolist())
        assert total_shared == 0, f"{total_shared} leaked subject-folds"
        assert len(covered) == len(index), "folds do not cover every segment"
        return "5 folds, 0 leaked subjects, full coverage"

    _check("subject-wise split is leak-free", _subject_split)

    def _segment_split():
        folds = splits.make_folds(index, scheme="segment", n_splits=5, seed=0)
        shared = sum(
            splits.leakage_report(index, tr, va)["n_shared_subjects"] for tr, va in folds
        )
        # This scheme is *supposed* to leak; assert it does, so the contrast
        # between the two protocols is demonstrated rather than asserted.
        assert shared > 0, "segment split unexpectedly leak-free"
        return f"{shared} shared subject-folds (expected -- this is the paper's setup)"

    _check("segment-wise split leaks (as expected)", _segment_split)

    def _holdout():
        tr, te = splits.make_holdout(index, scheme="subject", test_size=0.2, seed=0)
        r = splits.leakage_report(index, tr, te)
        assert r["n_shared_subjects"] == 0
        assert len(tr) + len(te) == len(index)
        return f"train {len(tr)} / test {len(te)} segments, 0 shared subjects"

    _check("subject holdout", _holdout)

    if not skip_torch:
        _check("torch available", _torch_check)

    return _summary()


def _torch_check():
    import torch

    from hyperppg.models.hybrid import PPGHybridClassifier
    from hyperppg.models.paper import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    notes = [f"torch {torch.__version__}, device {device}"]

    for name in ("alexnet", "vgg16", "avgpool_vgg16"):
        model = build_model(name, num_classes=4, pretrained=False)
        out = model(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, 4), (name, out.shape)
    notes.append("paper models forward ok")

    from hyperppg.data.preprocess import CLEAN_SEQ_LEN

    hybrid = PPGHybridClassifier(num_classes=4, n_tabular=0)
    out = hybrid(torch.zeros(2, 1, CLEAN_SEQ_LEN))
    assert out.shape == (2, 4), out.shape
    notes.append(f"hybrid forward ok (seq_len={CLEAN_SEQ_LEN})")

    return "; ".join(notes)


def _summary() -> int:
    print()
    print("-" * 72)
    n_fail = 0
    for name, ok, detail in _results:
        tag = _PASS if ok else _FAIL
        print(f"[{tag}] {name}")
        if detail:
            for line in str(detail).splitlines():
                print(f"          {line}")
        n_fail += not ok
    print("-" * 72)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="PPG-BP dataset root")
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args(argv)
    return run(args.root, skip_torch=args.skip_torch)


if __name__ == "__main__":
    sys.exit(main())
