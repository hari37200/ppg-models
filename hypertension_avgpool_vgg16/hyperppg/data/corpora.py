"""Unlabelled wrist-PPG corpora for self-supervised pretraining.

PPG-BP gives us 657 labelled segments, which is nowhere near enough to train a
transformer from scratch. The two datasets sitting next to this project are
unlabelled for our purposes but enormous by comparison, and both contain
Empatica E4 wrist BVP:

PPG-DaLiA    15 subjects x ~2.5 h. Crucially, the wrist BVP is available from
             ``SX/SX_E4.zip -> BVP.csv`` (~2.6 MB zipped per subject) rather
             than from ``SX.pkl`` (~1.4 GB per subject), because we only need
             the unlabelled signal, not the ECG-synchronised heart-rate labels.
             That is a ~500x saving and is what makes this step practical on a
             Colab/Kaggle disk quota.

FatigueSet   12 participants x 3 sessions, ``<pid>/<session>/wrist_bvp.csv``.

Both are sampled at 64 Hz. Windows are resampled to the same working rate as
PPG-BP so the pretrained encoder transfers without a domain shift in sampling.

Note the sensor domain gap: PPG-DaLiA and FatigueSet are *wrist* PPG under
daily-life motion, while PPG-BP is a clean seated *fingertip* recording.
Pretraining still helps -- pulse morphology is shared -- but do not expect the
gain you would get from a matched-site corpus.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from hyperppg.data.preprocess import bandpass, resample_to, zscore

__all__ = [
    "load_dalia_bvp",
    "load_fatigueset_bvp",
    "build_pretrain_windows",
    "E4_FS",
]

#: Empatica E4 BVP sampling rate (both corpora).
E4_FS = 64.0


def _read_e4_bvp_csv(handle) -> np.ndarray:
    """Parse an Empatica ``BVP.csv``.

    Line 1 is the start timestamp, line 2 the sampling rate, the rest samples.
    """
    text = handle.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return np.empty(0, dtype=np.float32)
    return np.asarray(lines[2:], dtype=np.float32)


def load_dalia_bvp(root: str | Path, verbose: bool = True) -> dict[str, np.ndarray]:
    """Read wrist BVP for every PPG-DaLiA subject from the E4 archives.

    ``root`` should be the directory containing ``S1/ ... S15/`` (i.e.
    ``.../PPG_FieldStudy``). Subjects whose archive is missing are skipped with
    a warning rather than failing the whole load.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PPG-DaLiA root not found: {root}")

    def _subject_dirs(base: Path) -> list[Path]:
        return sorted(
            (d for d in base.glob("S*") if d.is_dir() and d.name[1:].isdigit()),
            key=lambda d: int(d.name[1:]),
        )

    # Tolerate being handed a level above PPG_FieldStudy.
    subject_dirs = _subject_dirs(root)
    if not subject_dirs:
        for nested in ("PPG_FieldStudy", "data/PPG_FieldStudy"):
            cand = root / nested
            if cand.is_dir() and _subject_dirs(cand):
                root, subject_dirs = cand, _subject_dirs(cand)
                break
    if not subject_dirs:
        raise FileNotFoundError(f"no S<N> subject directories under {root}")

    out: dict[str, np.ndarray] = {}
    for sdir in subject_dirs:
        e4 = sdir / f"{sdir.name}_E4.zip"
        if not e4.is_file():
            if verbose:
                print(f"[dalia] skip {sdir.name}: no {e4.name}")
            continue
        try:
            with zipfile.ZipFile(e4) as zf:
                name = next(
                    (n for n in zf.namelist() if n.upper().endswith("BVP.CSV")), None
                )
                if name is None:
                    if verbose:
                        print(f"[dalia] skip {sdir.name}: no BVP.csv inside archive")
                    continue
                with zf.open(name) as fh:
                    sig = _read_e4_bvp_csv(io.TextIOWrapper(fh, encoding="utf-8"))
        except (zipfile.BadZipFile, OSError) as exc:
            if verbose:
                print(f"[dalia] skip {sdir.name}: {exc}")
            continue

        if sig.size:
            out[sdir.name] = sig
            if verbose:
                print(f"[dalia] {sdir.name}: {sig.size} samples ({sig.size/E4_FS/60:.1f} min)")
    if not out:
        raise RuntimeError(f"no BVP recovered from {root}")
    return out


def load_fatigueset_bvp(
    root: str | Path, verbose: bool = True
) -> dict[str, np.ndarray]:
    """Read ``<pid>/<session>/wrist_bvp.csv`` for every FatigueSet session.

    ``root`` is the directory containing the numbered participant folders
    (i.e. the extracted ``fatigueset/``). Accepts a ``.zip`` path too.
    """
    root = Path(root)
    out: dict[str, np.ndarray] = {}

    if root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith("wrist_bvp.csv"):
                    continue
                parts = Path(name).parts
                key = "_".join(parts[-3:-1]) if len(parts) >= 3 else name
                with zf.open(name) as fh:
                    sig = _read_csv_column(io.TextIOWrapper(fh, encoding="utf-8"))
                if sig.size:
                    out[key] = sig
                    if verbose:
                        print(f"[fatigueset] {key}: {sig.size} samples")
        if not out:
            raise RuntimeError(f"no wrist_bvp.csv found inside {root}")
        return out

    if not root.is_dir():
        raise FileNotFoundError(f"FatigueSet root not found: {root}")
    if (root / "fatigueset").is_dir():
        root = root / "fatigueset"

    for path in sorted(root.glob("*/*/wrist_bvp.csv")):
        key = f"{path.parent.parent.name}_{path.parent.name}"
        with path.open() as fh:
            sig = _read_csv_column(fh)
        if sig.size:
            out[key] = sig
            if verbose:
                print(f"[fatigueset] {key}: {sig.size} samples")
    if not out:
        raise RuntimeError(f"no wrist_bvp.csv found under {root}")
    return out


def _read_csv_column(handle, column: int = 1) -> np.ndarray:
    """Read a ``timestamp,value`` CSV with a header row into a value vector."""
    values: list[float] = []
    first = True
    for line in handle:
        if first:  # header
            first = False
            continue
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) <= column:
            continue
        try:
            values.append(float(parts[column]))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float32)


def build_pretrain_windows(
    recordings: dict[str, np.ndarray],
    fs_in: float = E4_FS,
    fs_out: float = 125.0,
    window_seconds: float = 2.1,
    stride_seconds: float = 1.0,
    max_windows: int | None = 200_000,
    flatness_percentile: float = 5.0,
    seed: int = 0,
    verbose: bool = True,
    window_len: int | None = None,
) -> np.ndarray:
    """Slice recordings into a ``(N, L)`` array of clean, normalised windows.

    Steps per recording: band-pass at the source rate -> resample to ``fs_out``
    -> slide a window -> drop the flattest windows (saturated or detached
    sensor) -> z-score each window.

    ``window_len`` overrides the sample count directly and defaults to
    :data:`hyperppg.data.preprocess.CLEAN_SEQ_LEN`, so pretraining windows are
    byte-for-byte the same length as a preprocessed PPG-BP segment. Deriving it
    from ``window_seconds * fs_out`` instead would give 262 rather than 263 --
    ``resample_poly`` rounds up -- and the mismatch would silently shift every
    positional encoding at fine-tuning time.
    """
    from hyperppg.data.preprocess import CLEAN_SEQ_LEN

    rng = np.random.default_rng(seed)
    win_len = int(window_len) if window_len is not None else int(CLEAN_SEQ_LEN)
    stride = max(int(round(stride_seconds * fs_out)), 1)

    chunks: list[np.ndarray] = []
    for key, sig in recordings.items():
        if sig.size < window_seconds * fs_in * 2:
            continue
        filtered = bandpass(sig, fs=fs_in, low=0.5, high=8.0, order=4)
        resampled = resample_to(filtered, fs_in=fs_in, fs_out=fs_out)
        n = resampled.size
        if n < win_len:
            continue
        starts = np.arange(0, n - win_len + 1, stride)
        if starts.size == 0:
            continue
        idx = starts[:, None] + np.arange(win_len)[None, :]
        w = resampled[idx]
        chunks.append(w.astype(np.float32))
        if verbose:
            print(f"[windows] {key}: {w.shape[0]} windows")

    if not chunks:
        raise RuntimeError("no windows produced -- recordings too short?")

    W = np.concatenate(chunks, axis=0)

    # Drop near-flat windows: no pulse means nothing to learn from.
    ptp = np.ptp(W, axis=1)
    if flatness_percentile > 0:
        thresh = np.percentile(ptp, flatness_percentile)
        keep = ptp > max(thresh, 1e-6)
        if verbose:
            print(f"[windows] dropping {int((~keep).sum())} flat windows")
        W = W[keep]

    if max_windows is not None and W.shape[0] > max_windows:
        sel = rng.choice(W.shape[0], size=max_windows, replace=False)
        W = W[np.sort(sel)]

    W = zscore(W)
    if verbose:
        print(f"[windows] final corpus: {W.shape}")
    return W
