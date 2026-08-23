"""Reading "any raw PPG signal" off disk.

Inference has to accept whatever a user actually has: one number per line, a
CSV with a header, an Empatica export, a NumPy array. This module normalises
all of that into ``(signal, fs)`` and is explicit about what it guessed, so a
wrong guess shows up in the output rather than in the results.

Supported
---------
``.npy``          a 1-D array (or a 2-D one, column picked by ``column``)
``.txt`` ``.dat`` whitespace- or newline-separated numbers, PPG-BP style
``.csv`` ``.tsv`` a table; the PPG column and, if present, a time column
``BVP.csv``       Empatica E4 export (start timestamp, then rate, then samples)
``.json``         ``[...]`` or ``{"signal": [...], "fs": 200}``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Recording", "load_signal", "expand_inputs", "SIGNAL_SUFFIXES"]

SIGNAL_SUFFIXES = (".csv", ".tsv", ".txt", ".dat", ".npy", ".json")

#: Column names that look like a PPG channel, in order of preference.
_PPG_NAMES = ("ppg", "bvp", "pleth", "plethysmogram", "signal", "wave",
              "waveform", "value", "amplitude", "ir", "red", "green")

#: Column names that look like a time base.
_TIME_NAMES = ("time", "timestamp", "t", "sec", "second", "seconds", "ms",
               "millis", "datetime", "elapsed")


@dataclass
class Recording:
    """A raw PPG signal plus everything known about where it came from."""

    signal: np.ndarray
    fs: float
    name: str
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.signal.size / self.fs)

    def describe(self) -> str:
        base = (f"{self.name}: {self.signal.size} samples at {self.fs:g} Hz "
                f"({self.duration:.1f} s)")
        return base + ("\n  " + "\n  ".join(self.notes) if self.notes else "")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _match(columns, names, min_substring=3):
    """First column matching one of ``names``, exact match preferred.

    Substring matching is only allowed for keys of at least ``min_substring``
    characters. Without that guard the time-column key "t" matches "mystery"
    and a perfectly good PPG column gets read as a clock.
    """
    lowered = {str(c): str(c).strip().lower() for c in columns}
    for want in names:
        for col, low in lowered.items():
            if low == want:
                return col
    for want in names:
        if len(want) < min_substring:
            continue
        for col, low in lowered.items():
            if want in low:
                return col
    return None


def _fs_from_time(t, notes):
    """Sampling rate from a time column, guessing its unit."""
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    d = np.diff(t)
    pos = d[d > 0]
    if pos.size < 2:
        return None
    fs = 1.0 / float(np.median(pos))

    # A time column rounded for display makes the median step a lie. PhysioNet's
    # BIDMC CSVs write two decimals, so a 0.008 s step is stored as 0.01 and
    # every sixth timestamp repeats -- the median says 100 Hz where the truth is
    # 125. The total span survives the rounding, so prefer it, but only when
    # repeated timestamps prove rounding is what happened: a recording with a
    # genuine gap has no ties, and there the median is the honest answer.
    ties = int((d == 0).sum())
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    if ties and span > 0 and not (d < 0).any():
        fs_span = (t.size - 1) / span
        if abs(fs_span - fs) > 0.01 * fs_span:
            notes.append(f"time column has {ties} repeated timestamps (rounded "
                         f"for display); rate read from the total span "
                         f"({fs_span:g} Hz), not the median step ({fs:g} Hz)")
            fs = fs_span

    if fs < 5.0:                      # the column was almost certainly in ms
        notes.append(f"time column looks like milliseconds; fs {fs:g} -> {fs*1000:g} Hz")
        fs *= 1000.0
    return round(fs, 6)


def _is_e4_export(path: Path) -> bool:
    """Empatica BVP.csv: a unix start time, then the rate, then bare samples."""
    try:
        with path.open() as fh:
            first, second = fh.readline().strip(), fh.readline().strip()
        return (float(first.split(",")[0]) > 1e8
                and 1.0 <= float(second.split(",")[0]) <= 2000.0)
    except (ValueError, OSError, IndexError):
        return False


def _read_e4(path: Path, notes):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    fs = float(lines[1].split(",")[0])
    sig = np.asarray([ln.split(",")[0] for ln in lines[2:]], dtype=np.float64)
    notes.append(f"Empatica E4 export; sampling rate {fs:g} Hz read from the header")
    return sig, fs


def _read_table(path: Path, column, notes):
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    df = pd.read_csv(path, sep=sep, engine="python")
    if df.empty:
        raise ValueError(f"{path} has no rows")

    # Headerless numeric file read as a table: the header became row zero.
    if all(str(c).replace(".", "", 1).replace("-", "", 1).isdigit()
           for c in df.columns):
        df = pd.read_csv(path, sep=sep, engine="python", header=None)
        notes.append("no header row detected; columns numbered from 0")

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError(f"{path} has no numeric columns")

    fs = None
    tcol = _match(df.columns, _TIME_NAMES)
    if tcol is None and numeric.shape[1] > 1:
        first = numeric.iloc[:, 0].to_numpy(dtype=float)
        if np.all(np.diff(first) > 0):            # monotonic -> a time base
            tcol = numeric.columns[0]
    if tcol is not None and tcol in numeric:
        fs = _fs_from_time(numeric[tcol], notes)
        if fs:
            notes.append(f"sampling rate {fs:g} Hz inferred from column '{tcol}'")

    if column is not None:
        col = (numeric.columns[int(column)] if str(column).lstrip("-").isdigit()
               else column)
        if col not in df.columns:
            raise KeyError(f"column {column!r} not in {list(df.columns)}")
    else:
        col = _match([c for c in numeric.columns if c != tcol], _PPG_NAMES)
        if col is None:
            candidates = [c for c in numeric.columns if c != tcol]
            if not candidates:
                raise ValueError(f"{path}: no signal column besides the time base")
            col = candidates[-1]
            notes.append(f"no obvious PPG column name; using '{col}' "
                         f"(pass --column to override)")
        else:
            notes.append(f"using column '{col}'")

    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float), fs


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------

def load_signal(path, fs=None, column=None, name=None) -> Recording:
    """Read a PPG recording from ``path``.

    ``fs`` always wins over anything found in the file. If neither is
    available the error says so rather than assuming a rate.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    notes, file_fs = [], None
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            idx = int(column) if column is not None else int(np.argmax(arr.std(axis=0)))
            notes.append(f"2-D array {arr.shape}; using column {idx}")
            arr = arr[:, idx]
        sig = np.asarray(arr, dtype=float).ravel()

    elif suffix == ".json":
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            key = next((k for k in ("signal", "ppg", "bvp", "data", "values")
                        if k in blob), None)
            if key is None:
                raise ValueError(f"{path}: no signal key in {list(blob)[:8]}")
            sig = np.asarray(blob[key], dtype=float).ravel()
            file_fs = blob.get("fs") or blob.get("sampling_rate") or blob.get("rate")
            if file_fs:
                notes.append(f"sampling rate {float(file_fs):g} Hz read from the file")
        else:
            sig = np.asarray(blob, dtype=float).ravel()

    elif suffix in (".csv", ".tsv") and _is_e4_export(path):
        sig, file_fs = _read_e4(path, notes)

    elif suffix in (".csv", ".tsv"):
        sig, file_fs = _read_table(path, column, notes)

    else:                                    # .txt, .dat, anything else textual
        text = path.read_text()
        sig = np.asarray(text.replace(",", " ").split(), dtype=float)
        notes.append(f"parsed {sig.size} whitespace-separated numbers")

    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size == 0:
        raise ValueError(f"{path}: no samples parsed")

    rate = float(fs) if fs else (float(file_fs) if file_fs else None)
    if rate is None:
        raise ValueError(
            f"{path}: sampling rate unknown. Pass --fs (e.g. --fs 200), or "
            f"supply a file with a time column, an Empatica header, or a JSON "
            f"'fs' field.")
    if fs and file_fs and abs(float(fs) - float(file_fs)) > 1e-6:
        notes.append(f"overriding the file's {float(file_fs):g} Hz with --fs {float(fs):g}")

    return Recording(signal=sig, fs=rate, name=name or path.stem,
                     source=str(path), notes=notes)


def expand_inputs(paths, pattern="*"):
    """Turn files, directories and globs into a sorted list of files."""
    out = []
    for p in ([paths] if isinstance(paths, (str, Path)) else list(paths)):
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.glob(pattern)
                              if q.is_file() and q.suffix.lower() in SIGNAL_SUFFIXES))
        elif p.is_file():
            out.append(p)
        else:                                   # treat as a glob pattern
            matches = sorted(Path().glob(str(p)))
            if not matches:
                raise FileNotFoundError(f"no input matches {p}")
            out.extend(m for m in matches if m.is_file())
    if not out:
        raise FileNotFoundError("no input files found")
    return out
