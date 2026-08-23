"""Hypertension stage for one PPG recording.

    python -m hyperppg.predict --input segment.txt
    python -m hyperppg.predict --from-dataset 2               # a real labelled subject
    python -m hyperppg.predict --input wrist.csv --fs 64 --json out.json

The model is written by :mod:`hyperppg.fit_model`. It expects the morphology of
a 2.1 s fingertip pulse: the recording is resampled to 1000 Hz, cut into 2.1 s
windows, scored per window, and the probabilities are averaged. A longer
recording therefore gets a steadier answer than a single window, which is the
only sense in which more data helps here.

What this is not
----------------
A subject-wise accuracy of ~0.49 on four classes is well above the 0.39
majority baseline and well below anything clinical. The printed verdict always
carries that number. Treat the output as a screening prior, not a diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hyperppg.config import FS_RAW, N_SAMPLES_RAW
from hyperppg.data.features import extract_features
from hyperppg.data.preprocess import bandpass, detrend, resample_to

DEFAULT_MODEL = Path("models") / "hypertension_hgb.joblib"

#: Column names that look like a PPG channel, best first.
_PPG_NAMES = ("ppg", "pleth", "bvp", "signal", "wave", "value", "ir", "red")
_TIME_NAMES = ("time", "timestamp", "sec", "seconds", "ms", "elapsed")


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def load_segment(path, fs=None, column=None):
    """Read one PPG recording. Returns ``(signal, fs, notes)``.

    Handles the PPG-BP native format (whitespace-separated numbers in a .txt),
    plus .csv/.tsv tables, .npy arrays and .json. The sampling rate comes from
    ``fs``, else a time column, else it defaults to PPG-BP's 1000 Hz -- and the
    default is always announced, because a wrong rate rescales every timing
    feature the model relies on.
    """
    path = Path(path)
    notes: list[str] = []
    suffix = path.suffix.lower()

    if suffix == ".npy":
        x = np.load(path).astype(np.float64).ravel()
    elif suffix == ".json":
        blob = json.loads(path.read_text())
        if isinstance(blob, dict):
            fs = fs or blob.get("fs")
            x = np.asarray(blob.get("signal", blob.get("ppg")), dtype=np.float64)
        else:
            x = np.asarray(blob, dtype=np.float64)
    elif suffix in (".csv", ".tsv"):
        import pandas as pd

        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        df.columns = [str(c).strip() for c in df.columns]
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError(f"{path} has no numeric columns")

        tcol = _match(numeric.columns, _TIME_NAMES)
        if fs is None and tcol is not None:
            fs = _fs_from_time(numeric[tcol].to_numpy(dtype=float), notes)
            if fs:
                notes.append(f"sampling rate {fs:g} Hz from column '{tcol}'")

        col = column or _match([c for c in numeric.columns if c != tcol],
                               _PPG_NAMES)
        if col is None:
            col = [c for c in numeric.columns if c != tcol][-1]
        if col not in numeric:
            raise KeyError(f"column {col!r} not in {list(numeric.columns)}")
        notes.append(f"using column '{col}'")
        x = numeric[col].to_numpy(dtype=np.float64)
    else:
        # .txt / .dat -- PPG-BP writes one whitespace-separated run of numbers.
        x = np.asarray(path.read_text().split(), dtype=np.float64)
        notes.append(f"parsed {x.size} whitespace-separated numbers")

    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"{path} contains no finite samples")

    if fs is None:
        fs = FS_RAW
        notes.append(f"no sampling rate given; assuming PPG-BP's {FS_RAW:g} Hz "
                     f"-- pass --fs if that is wrong")
    return x, float(fs), notes


def _fs_from_time(t, notes):
    """Sampling rate from a time column, tolerant of a rounded clock.

    PhysioNet's BIDMC CSVs write two decimals, so a 0.008 s step is stored as
    0.01 and every sixth timestamp repeats: the median step then says 100 Hz
    where the truth is 125, and every timing feature the model reads is
    rescaled by a quarter. The total span survives that rounding. Only prefer
    it when repeated timestamps prove rounding is what happened -- a recording
    with a genuine gap has no ties, and there the median is the honest answer.
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    d = np.diff(t)
    pos = d[d > 0]
    if pos.size < 2:
        return None
    fs = 1.0 / float(np.median(pos))

    ties = int((d == 0).sum())
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    if ties and span > 0 and not (d < 0).any():
        fs_span = (t.size - 1) / span
        if abs(fs_span - fs) > 0.01 * fs_span:
            notes.append(f"time column has {ties} repeated timestamps (rounded "
                         f"for display); rate read from the total span "
                         f"({fs_span:g} Hz), not the median step ({fs:g} Hz)")
            fs = fs_span
    return round(fs, 6)


def _match(columns, wanted):
    low = {str(c).lower(): c for c in columns}
    for want in wanted:
        if want in low:
            return low[want]
    for want in wanted:
        for lc, orig in low.items():
            if want in lc:
                return orig
    return None


def from_dataset(subject_id, root=None, segment=None):
    """Pull a real labelled segment out of PPG-BP.

    Returns ``(x, fs, label, key)`` where ``key`` is ``(subject_id, segment)``
    -- the handle :func:`oof_lookup` needs to find this segment's held-out
    score.
    """
    from hyperppg.data import ppgbp

    index = ppgbp.build_index(root)
    rows = index[index["subject_id"] == int(subject_id)]
    if rows.empty:
        raise SystemExit(
            f"subject {subject_id} is not in PPG-BP. Available ids run "
            f"{index['subject_id'].min()}-{index['subject_id'].max()}.")
    row = rows.iloc[0] if segment is None else rows[
        rows["segment"] == int(segment)].iloc[0]
    x = np.asarray(Path(row["path"]).read_text().split(), dtype=np.float64)
    return (x, FS_RAW, str(row["label"]),
            (int(row["subject_id"]), int(row["segment"])))


def signal_hash(x) -> str:
    """A stable fingerprint for one raw waveform.

    Rounded to two decimals before hashing so that a round-trip through a text
    file, which is how PPG-BP ships, still matches the array it was written
    from.
    """
    a = np.round(np.asarray(x, dtype=np.float64).ravel(), 2).astype(np.float32)
    return hashlib.sha1(a.tobytes()).hexdigest()


def oof_lookup(bundle, key=None, signal=None):
    """The held-out probabilities for a training segment, or ``None``.

    The deployed model was fitted on all 657 segments, so asking it about one
    of them returns a memorised answer -- typically 99%-confident and worth
    nothing. Every training segment does have exactly one honest score: the one
    it got from the fold that had never seen it. That is what this returns.

    Matches on the waveform itself when ``signal`` is given, so the check holds
    however the segment arrived, and falls back to ``(subject_id, segment)``.
    """
    table = bundle.get("oof")
    if not table:
        return None

    if signal is not None and table.get("hashes"):
        h = signal_hash(signal)
        hashes = table["hashes"]
        if h in hashes:
            return np.asarray(table["proba"][hashes.index(h)], dtype=np.float64)

    if key is not None:
        subj, seg = key
        hit = np.flatnonzero((table["subject_id"] == subj)
                             & (table["segment"] == seg))
        if hit.size:
            return np.asarray(table["proba"][hit[0]], dtype=np.float64)
    return None


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def predict(signal, fs, model, override_proba=None, detail=False):
    """Score one recording. Returns a result dict.

    ``override_proba`` substitutes a precomputed probability vector (the
    held-out one, for a segment the model was trained on) while still running
    the real pipeline, so the reported windowing and conditioning stay true.

    With ``detail=True`` returns ``(result, detail)``, the second carrying the
    conditioned windows and per-window probabilities that the report figure
    draws. Kept out of the result dict so it stays JSON-serialisable.
    """
    bundle = model
    names = bundle["feature_names"]
    classes = bundle["class_names"]
    target_fs = float(bundle["fs"])
    win = int(bundle["n_samples"])

    x = np.asarray(signal, dtype=np.float64).ravel()
    notes: list[str] = []

    if abs(fs - target_fs) > 1e-6:
        x = resample_to(x, fs_in=fs, fs_out=target_fs)
        notes.append(f"resampled {fs:g} Hz -> {target_fs:g} Hz")

    duration = x.size / target_fs
    if x.size < win:
        raise ValueError(
            f"{duration:.2f} s is too short. The model reads the morphology of "
            f"a {win / target_fs:.1f} s pulse train; give it at least that.")

    # Non-overlapping windows: a 2.1 s model applied honestly to a longer trace.
    n_win = x.size // win
    windows = x[: n_win * win].reshape(n_win, win)
    if n_win > 1:
        notes.append(f"{n_win} windows of {win / target_fs:.1f} s, "
                     f"probabilities averaged")

    conditioned = bandpass(detrend(windows), fs=target_fs, low=0.5, high=8.0)
    F = np.zeros((n_win, len(names)), dtype=np.float32)
    for i in range(n_win):
        feats = extract_features(conditioned[i], fs=target_fs)
        for j, name in enumerate(names):
            F[i, j] = feats.get(name, 0.0)

    proba = bundle["classifier"].predict_proba(F)
    if override_proba is not None:
        proba = np.asarray(override_proba, dtype=np.float64).reshape(1, -1)
        notes.append("this segment is in the training set; showing its "
                     "held-out (out-of-fold) score, not the fitted model's "
                     "memory of it")
    mean_proba = proba.mean(axis=0)
    order = np.argsort(mean_proba)[::-1]

    result = {
        "stage": classes[int(order[0])],
        "confidence": float(mean_proba[order[0]]),
        "probabilities": {classes[i]: float(mean_proba[i])
                          for i in range(len(classes))},
        "runner_up": classes[int(order[1])],
        "n_windows": int(n_win),
        "duration_s": float(duration),
        "fs": float(fs),
        "per_window": [classes[int(i)] for i in proba.argmax(axis=1)],
        "out_of_fold": override_proba is not None,
        "model_cv": bundle.get("cv", {}),
        "notes": notes,
    }
    if not detail:
        return result
    return result, {
        "windows": conditioned,
        # An out-of-fold override is one vector for the whole recording; the
        # figure still wants one row per window to draw.
        "proba": (np.repeat(proba, n_win, axis=0) if proba.shape[0] == 1
                  and n_win > 1 else proba),
        "class_names": classes,
        "fs": target_fs,
    }


def format_result(r, name="recording", truth=None) -> str:
    cv = r.get("model_cv", {})
    bar_w = 28
    lines = [
        f"  recording            {name}  ({r['duration_s']:.1f} s at {r['fs']:g} Hz)",
        f"  predicted stage      {r['stage']}",
        f"  confidence           {r['confidence']:.1%}   "
        f"(runner-up: {r['runner_up']})",
        "",
        "  class probabilities",
    ]
    for cls, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(p * bar_w))
        lines.append(f"    {cls:<24} {p:6.1%}  {bar}")

    if r["n_windows"] > 1:
        agree = sum(w == r["stage"] for w in r["per_window"])
        lines += ["", f"  windows              {r['n_windows']} scored, "
                      f"{agree} agree with the verdict"]

    if truth is not None:
        hit = "correct" if truth == r["stage"] else "WRONG"
        lines += ["", f"  ground truth         {truth}   -> {hit}"]

    if cv.get("accuracy"):
        lines += ["", f"  model accuracy       {cv['accuracy']:.1%} subject-wise "
                      f"out-of-fold (macro-F1 {cv.get('macro_f1', float('nan')):.3f})",
                  "                       majority-class baseline is 38.8%. This is a "
                  "screening",
                  "                       prior on 4 classes, not a diagnosis."]
    for n in r.get("notes", []):
        lines.append(f"  note: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hyperppg.predict",
        description="Hypertension stage from one PPG recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m hyperppg.predict --from-dataset 2
  python -m hyperppg.predict --input segment.txt
  python -m hyperppg.predict --input wrist.csv --fs 64 --json out.json
""")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="a PPG recording (.txt .csv .npy .json)")
    src.add_argument("--from-dataset", type=int, metavar="SUBJECT_ID",
                     help="score a real labelled PPG-BP subject and show the truth")
    ap.add_argument("--segment", type=int, help="which segment of that subject")
    ap.add_argument("--fs", type=float, help="sampling rate in Hz")
    ap.add_argument("--column", help="signal column name, for tables")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--root", default=None, help="PPG-BP root, for --from-dataset")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--plot", help="write a one-page report figure here (.png)")
    ap.add_argument("--quiet", "-q", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"no model at {model_path}. Build one first:\n"
              f"    python -m hyperppg.fit_model", file=sys.stderr)
        return 2

    import joblib

    bundle = joblib.load(model_path)
    if bundle.get("schema") != 1:
        print(f"warning: {model_path} has schema {bundle.get('schema')}, "
              f"expected 1", file=sys.stderr)

    truth, key = None, None
    if args.from_dataset is not None:
        x, fs, truth, key = from_dataset(args.from_dataset, args.root,
                                         args.segment)
        name = f"PPG-BP subject {args.from_dataset}"
        notes = []
    else:
        x, fs, notes = load_segment(args.input, fs=args.fs, column=args.column)
        name = Path(args.input).stem

    # Checked for every route, not just --from-dataset: a training segment
    # handed back as a plain file must not come out looking 99% certain.
    override = oof_lookup(bundle, key=key, signal=x)

    if not args.quiet:
        print(f"model: {model_path}   {bundle['model']}, trained on "
              f"{bundle['n_train_segments']} segments from "
              f"{bundle['n_train_subjects']} subjects\n")

    out = predict(x, fs, bundle, override_proba=override,
                  detail=bool(args.plot))
    result, detail = out if args.plot else (out, None)
    result["notes"] = notes + result["notes"]
    result["name"] = name
    if truth is not None:
        result["ground_truth"] = truth

    if not args.quiet:
        print(format_result(result, name=name, truth=truth))

    if args.plot:
        from hyperppg.plotting import plot_report

        dest = Path(args.plot)
        dest.parent.mkdir(parents=True, exist_ok=True)
        plot_report(result, detail, path=dest)
        print(f"\nwrote {dest}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
