"""The three public corpora, and finding them wherever they were unpacked.

None of these carry the paper's own measurements -- that data is available only
"from the corresponding author on reasonable request". What they give you:

PPG-BP       657 clean seated fingertip segments (2.1 s at 1000 Hz) from 219
             subjects with clinical records. No fatigue labels. Validates the
             peak detector on real signals and shows what the index does on a
             cohort that is not sixteen healthy 22-year-olds. 1.5 MB, downloads
             itself.
FatigueSet   12 participants x 3 sessions of wrist BVP at 64 Hz *with fatigue
             self-reports*. The closest public analogue to the paper's design.
PPG-DaLiA    15 subjects x ~2.5 h of wrist BVP through a daily-life protocol.
             No fatigue labels; used for stability checks.
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DALIA_MARKER, E4_FS, FATIGUESET_MARKER, FIGSHARE_PPGBP,
                     PPGBP_MARKER, PPGBP_N_SEGMENTS, PPGBP_N_SUBJECTS,
                     default_data_dir)

__all__ = ["find_root", "resolve_all", "download_ppgbp", "load_ppgbp",
           "load_dalia", "load_fatigueset", "discover_tables"]

SEARCH_BASES = ("/kaggle/input", "/kaggle/working", "/content", "data", ".")


def find_root(marker, env_var=None, extra=(), bases=SEARCH_BASES, max_depth=3):
    """First directory (breadth-first) containing ``marker``, else None.

    An environment variable always wins, so a dataset in an unusual place needs
    no code change: ``FATIGUESET_ROOT=/mnt/data/fatigueset``.
    """
    probes = [Path(p) for p in extra]
    if env_var and os.environ.get(env_var):
        probes.insert(0, Path(os.environ[env_var]))
    probes.append(default_data_dir())
    for p in probes:
        if p.is_dir() and next(p.glob(marker), None) is not None:
            return p.resolve()

    for base in bases:
        base = Path(base)
        if not base.is_dir():
            continue
        level, depth = [base], 0
        while level and depth <= max_depth:
            nxt = []
            for d in level:
                try:
                    if next(d.glob(marker), None) is not None:
                        return d.resolve()
                    nxt.extend(c for c in sorted(d.iterdir()) if c.is_dir())
                except (OSError, PermissionError):
                    continue
            level, depth = nxt, depth + 1
    return None


def resolve_all(verbose=True) -> dict:
    """Locate all three corpora. Missing ones come back as None."""
    roots = {
        "ppgbp": find_root(PPGBP_MARKER, "PPGBP_ROOT"),
        "fatigueset": find_root(FATIGUESET_MARKER, "FATIGUESET_ROOT"),
        "dalia": find_root(DALIA_MARKER, "DALIA_ROOT"),
    }
    if verbose:
        for name, root in roots.items():
            print(f"  {name:<11} {root if root else 'not found'}")
    return roots


# --------------------------------------------------------------------------
# PPG-BP
# --------------------------------------------------------------------------

def download_ppgbp(dest=None, force=False) -> Path:
    """Fetch and extract PPG-BP (1.5 MB) so ``dest/Data File/0_subject`` exists."""
    dest = Path(dest) if dest else default_data_dir() / "ppgbp"
    if next(dest.glob(PPGBP_MARKER), None) is not None and not force:
        print(f"[ppg-bp] already present at {dest.resolve()}")
        return dest.resolve()

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[ppg-bp] fetching {FIGSHARE_PPGBP}")
    with urllib.request.urlopen(FIGSHARE_PPGBP, timeout=300) as r:
        blob = r.read()
    print(f"[ppg-bp] got {len(blob)/1e6:.2f} MB")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(tmp)
        root = next((c.parent.parent for c in tmp.rglob("0_subject") if c.is_dir()), None)
        if root is None:
            raise RuntimeError("unexpected archive layout: no 0_subject directory")
        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))

    n = len(list((dest / "Data File" / "0_subject").glob("*.txt")))
    print(f"[ppg-bp] {n} segments (expected {PPGBP_N_SEGMENTS}) at {dest.resolve()}")
    return dest.resolve()


def load_ppgbp(root):
    """-> ``(index DataFrame, signals (N, 2100) float32 at 1000 Hz)``.

    The spreadsheet is per subject and the signals are per segment, so each
    subject's clinical row is broadcast onto that subject's three segments.
    """
    root = Path(root)
    df = pd.read_excel(root / "Data File" / "PPG-BP dataset.xlsx",
                       sheet_name="cardiovascular dataset", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    clin = pd.DataFrame({
        "subject_id": pd.to_numeric(df["subject_ID"], errors="coerce"),
        "label": df["Hypertension"].astype(str).str.strip(),
        "age": pd.to_numeric(df.get("Age(year)"), errors="coerce"),
        "sex": df["Sex(M/F)"].astype(str).str.strip().str.lower(),
        "bmi": pd.to_numeric(df.get("BMI(kg/m^2)"), errors="coerce"),
        "sbp": pd.to_numeric(df.get("Systolic Blood Pressure(mmHg)"), errors="coerce"),
        "dbp": pd.to_numeric(df.get("Diastolic Blood Pressure(mmHg)"), errors="coerce"),
        "hr_ref": pd.to_numeric(df.get("Heart Rate(b/m)"), errors="coerce"),
    }).dropna(subset=["subject_id"])
    clin["subject_id"] = clin["subject_id"].astype(int)

    rows = []
    for p in sorted((root / "Data File" / "0_subject").glob("*.txt")):
        sid, _, seg = p.stem.rpartition("_")
        try:
            rows.append({"subject_id": int(sid), "segment": int(seg), "path": str(p)})
        except ValueError:
            continue
    index = (pd.DataFrame(rows).merge(clin, on="subject_id", how="inner")
             .sort_values(["subject_id", "segment"]).reset_index(drop=True))
    if len(index) != PPGBP_N_SEGMENTS or index["subject_id"].nunique() != PPGBP_N_SUBJECTS:
        print(f"[ppg-bp] WARNING: {len(index)} segments / "
              f"{index['subject_id'].nunique()} subjects, expected "
              f"{PPGBP_N_SEGMENTS} / {PPGBP_N_SUBJECTS}")

    sig = np.zeros((len(index), 2100), dtype=np.float32)
    for i, p in enumerate(index["path"]):
        v = np.asarray(Path(p).read_text().split(), dtype=np.float32)
        n = min(v.size, 2100)
        sig[i, :n] = v[:n]
        if n < 2100:
            sig[i, n:] = v[-1] if n else 0.0
    return index, sig


# --------------------------------------------------------------------------
# Empatica corpora
# --------------------------------------------------------------------------

def _read_e4_bvp(handle):
    text = handle.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (np.asarray(lines[2:], dtype=np.float32) if len(lines) > 2
            else np.empty(0, dtype=np.float32))


def _read_value_column(handle, column=1):
    vals, first = [], True
    for line in handle:
        if first:
            first = False
            continue
        parts = line.strip().split(",")
        if len(parts) > column:
            try:
                vals.append(float(parts[column]))
            except ValueError:
                pass
    return np.asarray(vals, dtype=np.float32)


def load_dalia(root, verbose=True) -> dict:
    """``{subject: BVP at 64 Hz}``, read straight out of each ``SX_E4.zip``.

    The wrist BVP is in the 2.6 MB E4 archive, not the 1.4 GB ``SX.pkl`` --
    that one carries the ECG-synchronised heart-rate labels, which are not
    needed here.
    """
    out = {}
    for sdir in sorted((d for d in Path(root).glob("S*") if d.is_dir()),
                       key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0):
        e4 = sdir / f"{sdir.name}_E4.zip"
        if not e4.is_file():
            continue
        try:
            with zipfile.ZipFile(e4) as zf:
                name = next((n for n in zf.namelist()
                             if n.upper().endswith("BVP.CSV")), None)
                if name is None:
                    continue
                with zf.open(name) as fh:
                    sig = _read_e4_bvp(io.TextIOWrapper(fh, encoding="utf-8"))
        except (zipfile.BadZipFile, OSError) as exc:
            if verbose:
                print(f"[dalia] skip {sdir.name}: {exc}")
            continue
        if sig.size:
            out[sdir.name] = sig
            if verbose:
                print(f"[dalia] {sdir.name}: {sig.size/E4_FS/60:.1f} min")
    return out


def load_fatigueset(root, verbose=True) -> dict:
    """``{(participant, session): BVP at 64 Hz}``."""
    out = {}
    for p in sorted(Path(root).glob("*/*/wrist_bvp.csv")):
        with p.open() as fh:
            sig = _read_value_column(fh)
        if sig.size:
            key = (p.parent.parent.name, p.parent.name)
            out[key] = sig
            if verbose:
                print(f"[fatigueset] {key[0]}/{key[1]}: {sig.size/E4_FS/60:.1f} min")
    return out


# --------------------------------------------------------------------------
# Finding self-report tables in a corpus whose schema you do not know
# --------------------------------------------------------------------------

FATIGUE_KEYS = ("fatigue", "tired", "exhaust", "exert", "rpe", "borg", "vas",
                "kss", "sleepi", "sleepy", "energy", "effort", "nasa", "tlx",
                "demand", "frustrat", "drowsi", "alert")
ID_KEYS = ("participant", "pid", "subject", "user", "person", "id")
SESSION_KEYS = ("session", "condition", "block", "visit", "intensity", "trial",
                "activity", "round", "phase")
SENSOR_STEMS = ("wrist_", "chest_", "muse_", "earbud_", "head_", "eeg", "acc",
                "bvp", "eda", "temp", "ibi", "hr", "tags", "gyro", "ppg")


def read_tables(path):
    """``[(sheet, DataFrame)]`` for a csv, or one entry per worksheet."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return [("", pd.read_csv(path))]
    return list(pd.read_excel(path, sheet_name=None).items())


def match_columns(columns, keys):
    return [c for c in columns if any(k in str(c).lower() for k in keys)]


def discover_tables(root, max_mb=5.0) -> pd.DataFrame:
    """Every non-sensor table under ``root``, ranked by fatigue-like columns.

    FatigueSet's survey schema is not documented anywhere this package can
    rely on, so rather than hard-code column names it reports what is actually
    there and lets the caller confirm.
    """
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in (".csv", ".xlsx", ".xls") or not p.is_file():
            continue
        if p.stat().st_size > max_mb * 1e6:
            continue
        if any(p.stem.lower().startswith(s) for s in SENSOR_STEMS):
            continue
        try:
            tables = read_tables(p)
        except Exception as exc:
            rows.append(dict(path=str(p), sheet="", n_rows=0,
                             error=type(exc).__name__))
            continue
        for sheet, df in tables:
            hits = match_columns(df.columns, FATIGUE_KEYS)
            rows.append(dict(
                path=str(p), sheet=sheet, n_rows=len(df), n_cols=len(df.columns),
                n_fatigue_cols=len(hits),
                id_cols=", ".join(map(str, match_columns(df.columns, ID_KEYS)[:3])),
                session_cols=", ".join(map(str, match_columns(df.columns, SESSION_KEYS)[:3])),
                fatigue_cols=", ".join(map(str, hits[:8])),
                all_cols=", ".join(map(str, df.columns[:25]))))
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("n_fatigue_cols", ascending=False)
            .reset_index(drop=True))
