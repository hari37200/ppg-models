"""Feature extraction: many recordings in, one table out.

This is the step between raw signals and training. It runs the same
Section 3.3-3.6 pipeline as inference, but over a whole cohort, and carries any
label columns straight through so ``fatigueppg.train`` can fit against them.

    python -m fatigueppg.extract --manifest cohort.csv -o features.csv
    python -m fatigueppg.extract --dataset ppgbp -o ppgbp_features.csv
    python -m fatigueppg.extract --input recordings/ --fs 200 -o features.csv

A manifest is a CSV with a ``path`` column and any of:

    fs        sampling rate for that file (else --fs, else read from the file)
    subject   grouping key; several recordings per subject are averaged at
              training time so one person cannot count as five
    session   free-text label carried through
    score     the subjective fatigue state, if you already have it
    q1..q9    BFI-Taiwan answers; training can average the paper's items 2 and 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyse_ppg, window_analysis
from .config import DURATION_PAPER, E4_FS, MIN_SQI, PPGBP_FS, ZERO_FRAC
from .hrv import hrv_indices, rr_series
from .quality import choose_orientation
from .signals import expand_inputs, load_signal

__all__ = ["extract_one", "extract_manifest", "extract_corpus", "main"]

PASSTHROUGH = ("subject", "session", "score", "label", "group") + tuple(
    f"q{i}" for i in range(1, 10))


def extract_one(signal, fs, name="rec", zero_frac=ZERO_FRAC, invert="auto",
                **extra) -> dict:
    """Features for one recording: the index, the HRV indices, and quality."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    inverted = False
    if invert == "auto":
        x, inverted, _, _ = choose_orientation(
            x, fs, lambda s, f: analyse_ppg(s, f, zero_frac=zero_frac))
    elif invert in (True, "yes"):
        x, inverted = -x, True

    res = analyse_ppg(x, fs, zero_frac=zero_frac)
    hrv = hrv_indices(*rr_series(res.peaks, fs))
    row = dict(name=name, **extra)
    row.update(res.summary())
    row.update({k: hrv[k] for k in ("nlf", "nhf", "nhf_0_10", "lf_hf",
                                    "sdnn", "rmssd", "n_rr")})
    row["inverted"] = inverted
    ratios = res.beats["ratio"].to_numpy(dtype=float) if len(res.beats) else np.array([])
    ratios = ratios[np.isfinite(ratios)]
    row["ratio_median"] = float(np.median(ratios)) if ratios.size else np.nan
    return row


def _iter_manifest(manifest_path, default_fs=None):
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    if "path" not in df.columns:
        raise KeyError(f"{manifest_path} needs a 'path' column; "
                       f"found {list(df.columns)}")
    base = manifest_path.parent
    for _, row in df.iterrows():
        p = Path(str(row["path"]))
        if not p.is_absolute():
            p = base / p
        fs = row.get("fs") if "fs" in df.columns else None
        fs = float(fs) if fs is not None and np.isfinite(pd.to_numeric(fs, errors="coerce")) else default_fs
        extra = {k: row[k] for k in PASSTHROUGH if k in df.columns}
        yield p, fs, extra


def extract_manifest(manifest_path, default_fs=None, column=None,
                     zero_frac=ZERO_FRAC, invert="auto", progress=True):
    rows, failures = [], []
    for path, fs, extra in _iter_manifest(manifest_path, default_fs):
        try:
            rec = load_signal(path, fs=fs, column=column)
            rows.append(extract_one(rec.signal, rec.fs, name=rec.name,
                                    zero_frac=zero_frac, invert=invert,
                                    path=str(path), **extra))
            if progress:
                print(f"  {rec.name:<28} index {rows[-1]['fatigue_index']:6.3f}  "
                      f"SQI {rows[-1]['sqi']:.2f}")
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            print(f"  [skip] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return pd.DataFrame(rows), failures


def extract_files(paths, fs=None, column=None, zero_frac=ZERO_FRAC,
                  invert="auto", progress=True):
    rows, failures = [], []
    for path in paths:
        try:
            rec = load_signal(path, fs=fs, column=column)
            rows.append(extract_one(rec.signal, rec.fs, name=rec.name,
                                    zero_frac=zero_frac, invert=invert,
                                    path=str(path)))
            if progress:
                print(f"  {rec.name:<28} index {rows[-1]['fatigue_index']:6.3f}  "
                      f"SQI {rows[-1]['sqi']:.2f}")
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            print(f"  [skip] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return pd.DataFrame(rows), failures


def extract_corpus(which, root=None, zero_frac=ZERO_FRAC, window_s=DURATION_PAPER,
                   max_windows=None, max_records=None, min_sqi=0.0):
    """Features for one of the three public corpora.

    PPG-BP gives one row per 2.1 s segment; the wrist corpora give one row per
    2-minute window, which is the paper's measurement length.
    """
    from . import datasets as ds

    if which == "ppgbp":
        root = root or ds.find_root(ds.PPGBP_MARKER if hasattr(ds, "PPGBP_MARKER")
                                    else "Data File/0_subject", "PPGBP_ROOT")
        if root is None:
            root = ds.download_ppgbp()
        index, signals = ds.load_ppgbp(root)
        if max_records:
            index, signals = index.iloc[:max_records], signals[:max_records]
        rows = []
        for i in range(len(index)):
            meta = index.iloc[i]
            try:
                rows.append(extract_one(
                    signals[i], PPGBP_FS, name=f"{meta['subject_id']}_{meta['segment']}",
                    zero_frac=zero_frac, invert="no",
                    subject=str(meta["subject_id"]), session=str(meta["segment"]),
                    label=meta["label"], age=meta["age"], sbp=meta["sbp"],
                    dbp=meta["dbp"], bmi=meta["bmi"], hr_ref=meta["hr_ref"]))
            except Exception as exc:
                print(f"  [skip] segment {meta['subject_id']}_{meta['segment']}: "
                      f"{type(exc).__name__}", file=sys.stderr)
        return pd.DataFrame(rows)

    if which in ("fatigueset", "dalia"):
        marker = ds.FATIGUESET_MARKER if which == "fatigueset" else ds.DALIA_MARKER
        env = "FATIGUESET_ROOT" if which == "fatigueset" else "DALIA_ROOT"
        root = root or ds.find_root(marker, env)
        if root is None:
            raise FileNotFoundError(
                f"{which} not found. See the dataset section of the README, or "
                f"set {env}=/path/to/{which}")
        recordings = (ds.load_fatigueset(root) if which == "fatigueset"
                      else ds.load_dalia(root))
        frames = []
        for key, sig in list(recordings.items())[:max_records]:
            subject, session = key if isinstance(key, tuple) else (key, "all")
            frames.append(window_analysis(
                sig, E4_FS, window_s=window_s, max_windows=max_windows,
                min_sqi=min_sqi, zero_frac=zero_frac,
                subject=str(subject), session=str(session)))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(out):
            out["name"] = out["subject"] + "_" + out["session"] + "_w" + out["window"].astype(str)
        return out

    raise ValueError(f"unknown corpus {which!r}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="fatigue-extract",
        description="Run the fatigue-index pipeline over many recordings.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="CSV with a 'path' column (see module docs)")
    src.add_argument("--input", nargs="+", help="files, a directory, or a glob")
    src.add_argument("--dataset", choices=["ppgbp", "fatigueset", "dalia"],
                     help="one of the public corpora")
    p.add_argument("--root", help="dataset root, if auto-detection fails")
    p.add_argument("--glob", default="*", help="pattern used inside a directory")
    p.add_argument("--fs", type=float, help="sampling rate for files without one")
    p.add_argument("--column", help="signal column name or index, for tables")
    p.add_argument("--zero-frac", type=float, default=ZERO_FRAC,
                   help="index zero point (0.5 = the paper)")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--window", type=float, default=DURATION_PAPER,
                   help="window length in seconds, for the long wrist corpora")
    p.add_argument("--max-windows", type=int, help="cap windows per recording")
    p.add_argument("--max-records", type=int, help="cap recordings (for a quick pass)")
    p.add_argument("--min-sqi", type=float, default=0.0,
                   help=f"drop rows below this signal quality (suggest {MIN_SQI})")
    p.add_argument("--out", "-o", required=True, help="output features CSV")
    args = p.parse_args(argv)

    if args.manifest:
        table, failures = extract_manifest(args.manifest, args.fs, args.column,
                                           args.zero_frac, args.invert)
    elif args.input:
        table, failures = extract_files(expand_inputs(args.input, args.glob),
                                        args.fs, args.column, args.zero_frac,
                                        args.invert)
    else:
        table, failures = extract_corpus(
            args.dataset, root=args.root, zero_frac=args.zero_frac,
            window_s=args.window, max_windows=args.max_windows,
            max_records=args.max_records, min_sqi=args.min_sqi), []

    if not len(table):
        print("no recording produced features", file=sys.stderr)
        return 1
    if args.min_sqi > 0 and "sqi" in table:
        before = len(table)
        table = table[table["sqi"] >= args.min_sqi].reset_index(drop=True)
        print(f"kept {len(table)}/{before} rows above SQI {args.min_sqi}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\n{len(table)} rows -> {args.out}")
    cols = [c for c in ("fatigue_index", "fatigue_index_onset", "hr", "nhf", "sqi")
            if c in table]
    print(table[cols].describe().round(3).to_string())
    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for path, why in failures[:10]:
            print(f"  {path}: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
