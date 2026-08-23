"""Build the PPG-BP segment index and load raw waveforms.

The dataset ships as:
    Data File/0_subject/<subject_ID>_<segment>.txt   657 files, 2100 tab-separated
                                                     floats each (2.1 s @ 1000 Hz)
    Data File/PPG-BP dataset.xlsx                    one row per subject

The spreadsheet is per *subject* while the signals are per *segment*, so each
subject's clinical row is broadcast onto that subject's three segments. This is
also exactly why a segment-level random split leaks: the same subject (and hence
the same label and the same clinical state) lands on both sides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyperppg.config import (
    COL_AGE,
    COL_BMI,
    COL_DBP,
    COL_DIABETES,
    COL_HEIGHT,
    COL_HR,
    COL_LABEL,
    COL_SBP,
    COL_SEX,
    COL_SUBJECT,
    COL_WEIGHT,
    EXPECTED_N_SEGMENTS,
    EXPECTED_N_SUBJECTS,
    LABEL_TO_IDX,
    N_SAMPLES_RAW,
    SUBJECT_DIR_REL,
    XLSX_HEADER_ROW,
    XLSX_REL,
    XLSX_SHEET,
    resolve_ppgbp_root,
)

#: Numeric clinical columns carried through to the index (used by the tabular
#: fusion head, and available for analysis).
TABULAR_COLS = ("age", "sex", "height", "weight", "bmi", "hr", "diabetes")


def _read_clinical(root: Path) -> pd.DataFrame:
    """Read the spreadsheet into a tidy per-subject frame."""
    xlsx = root / XLSX_REL
    df = pd.read_excel(xlsx, sheet_name=XLSX_SHEET, header=XLSX_HEADER_ROW)

    # Column names occasionally carry stray whitespace across pandas versions.
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in (COL_SUBJECT, COL_LABEL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{xlsx} is missing expected column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["subject_id"] = pd.to_numeric(df[COL_SUBJECT], errors="coerce").astype("Int64")
    out["label"] = df[COL_LABEL].astype(str).str.strip()

    out["age"] = pd.to_numeric(df.get(COL_AGE), errors="coerce")
    out["height"] = pd.to_numeric(df.get(COL_HEIGHT), errors="coerce")
    out["weight"] = pd.to_numeric(df.get(COL_WEIGHT), errors="coerce")
    out["bmi"] = pd.to_numeric(df.get(COL_BMI), errors="coerce")
    out["sbp"] = pd.to_numeric(df.get(COL_SBP), errors="coerce")
    out["dbp"] = pd.to_numeric(df.get(COL_DBP), errors="coerce")
    out["hr"] = pd.to_numeric(df.get(COL_HR), errors="coerce")

    sex = df.get(COL_SEX)
    out["sex"] = (
        sex.astype(str).str.strip().str.lower().map({"male": 1.0, "female": 0.0})
        if sex is not None
        else np.nan
    )

    diabetes = df.get(COL_DIABETES)
    out["diabetes"] = (
        diabetes.notna().astype(float) if diabetes is not None else 0.0
    )

    # Drop the banner/blank rows and anything without a usable label.
    out = out[out["subject_id"].notna()]
    out = out[out["label"].isin(LABEL_TO_IDX)]
    out = out.reset_index(drop=True)
    out["subject_id"] = out["subject_id"].astype(int)
    return out


def build_index(root: str | Path | None = None, strict: bool = True) -> pd.DataFrame:
    """Return one row per PPG segment, joined with its subject's clinical data.

    Parameters
    ----------
    root
        Dataset root (the directory containing ``Data File``). Auto-detected
        when omitted.
    strict
        Raise if the resulting index does not match the published 657
        segments / 219 subjects. Set False to tolerate a partial copy.

    Returns
    -------
    DataFrame with columns:
        ``subject_id, segment, path, label, y`` + :data:`TABULAR_COLS` + sbp/dbp
    """
    root = resolve_ppgbp_root(root)
    clinical = _read_clinical(root)

    subj_dir = root / SUBJECT_DIR_REL
    rows = []
    for path in sorted(subj_dir.glob("*.txt")):
        stem = path.stem  # "<subject>_<segment>"
        if "_" not in stem:
            continue
        sid_str, _, seg_str = stem.rpartition("_")
        try:
            rows.append(
                {
                    "subject_id": int(sid_str),
                    "segment": int(seg_str),
                    "path": str(path),
                }
            )
        except ValueError:
            continue

    if not rows:
        raise FileNotFoundError(f"no <subject>_<segment>.txt files under {subj_dir}")

    seg = pd.DataFrame(rows)
    index = seg.merge(clinical, on="subject_id", how="inner")
    index = index.sort_values(["subject_id", "segment"]).reset_index(drop=True)
    index["y"] = index["label"].map(LABEL_TO_IDX).astype(int)

    n_seg, n_sub = len(index), index["subject_id"].nunique()
    if strict and (n_seg != EXPECTED_N_SEGMENTS or n_sub != EXPECTED_N_SUBJECTS):
        raise RuntimeError(
            f"index has {n_seg} segments / {n_sub} subjects, expected "
            f"{EXPECTED_N_SEGMENTS} / {EXPECTED_N_SUBJECTS}. "
            f"Signal files found: {len(seg)}, clinical rows: {len(clinical)}. "
            "Pass strict=False to continue anyway."
        )
    return index


def _read_segment(path: str | Path) -> np.ndarray:
    """Read one .txt segment into a float32 vector."""
    text = Path(path).read_text()
    arr = np.asarray(text.split(), dtype=np.float32)
    return arr


def load_signals(
    index: pd.DataFrame,
    cache: str | Path | None = None,
    n_samples: int = N_SAMPLES_RAW,
) -> np.ndarray:
    """Load every segment referenced by ``index`` into an ``(N, n_samples)`` array.

    Segments whose length differs from ``n_samples`` are right-padded with their
    last value or truncated, so the output is always rectangular.

    ``cache`` optionally points at a ``.npy`` file. It is only reused when its
    shape matches ``index``; otherwise it is silently rewritten.
    """
    if cache is not None:
        cache = Path(cache)
        if cache.is_file():
            arr = np.load(cache)
            if arr.shape == (len(index), n_samples):
                return arr

    out = np.empty((len(index), n_samples), dtype=np.float32)
    for i, path in enumerate(index["path"].tolist()):
        sig = _read_segment(path)
        if sig.size == n_samples:
            out[i] = sig
        elif sig.size > n_samples:
            out[i] = sig[:n_samples]
        else:
            out[i, : sig.size] = sig
            out[i, sig.size :] = sig[-1] if sig.size else 0.0

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def load_dataset(
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convenience wrapper: ``(signals, y, index)``."""
    index = build_index(root, strict=strict)
    cache = Path(cache_dir) / "ppgbp_raw.npy" if cache_dir else None
    signals = load_signals(index, cache=cache)
    y = index["y"].to_numpy(dtype=np.int64)
    return signals, y, index


def summarise(index: pd.DataFrame) -> str:
    """Human-readable class balance at both segment and subject level."""
    lines = [
        f"segments: {len(index)}   subjects: {index['subject_id'].nunique()}",
        "",
        f"{'class':<24} {'segments':>9} {'subjects':>9}",
    ]
    per_subject = index.drop_duplicates("subject_id")
    for name, idx in LABEL_TO_IDX.items():
        n_seg = int((index["label"] == name).sum())
        n_sub = int((per_subject["label"] == name).sum())
        lines.append(f"{name:<24} {n_seg:>9} {n_sub:>9}")

    maj_seg = index["label"].value_counts().iloc[0] / len(index)
    maj_sub = per_subject["label"].value_counts().iloc[0] / len(per_subject)
    lines += [
        "",
        f"majority-class baseline -- segment level: {maj_seg:.3f}, "
        f"subject level: {maj_sub:.3f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    idx = build_index()
    print(summarise(idx))
