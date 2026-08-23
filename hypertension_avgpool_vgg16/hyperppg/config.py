"""Constants and filesystem-layout resolution.

Everything that depends on *where* the data lives is funnelled through
``resolve_ppgbp_root`` so the same code runs unchanged on a local machine,
Google Colab and Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Dataset facts (verified against the actual figshare archive, not assumed)
# --------------------------------------------------------------------------

#: Raw sampling rate of the PPG-BP recordings. Each .txt holds 2100 samples
#: spanning 2.1 s, i.e. exactly 1000 Hz.
FS_RAW = 1000.0

#: Number of samples in every PPG-BP segment file.
N_SAMPLES_RAW = 2100

#: The four label strings exactly as they appear in the "Hypertension" column
#: of "PPG-BP dataset.xlsx". Order defines the integer class index.
CLASS_NAMES = (
    "Normal",
    "Prehypertension",
    "Stage 1 hypertension",
    "Stage 2 hypertension",
)

#: Short aliases used in the paper's tables (nt / pt / ht1 / ht2).
CLASS_ALIASES = ("nt", "pt", "ht1", "ht2")

NUM_CLASSES = len(CLASS_NAMES)

LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}

#: Subject-level class counts in the released dataset, for sanity checking.
#: (Normal 80, Prehypertension 85, Stage 1 34, Stage 2 20 -> 219 subjects.)
EXPECTED_SUBJECT_COUNTS = {
    "Normal": 80,
    "Prehypertension": 85,
    "Stage 1 hypertension": 34,
    "Stage 2 hypertension": 20,
}

EXPECTED_N_SUBJECTS = 219
EXPECTED_N_SEGMENTS = 657

#: figshare file id for "PPG-BP Database.zip" (article 5459299), ~1.5 MB.
FIGSHARE_URL = "https://ndownloader.figshare.com/files/9441097"

#: Relative layout inside the extracted archive.
SUBJECT_DIR_REL = Path("Data File") / "0_subject"
XLSX_REL = Path("Data File") / "PPG-BP dataset.xlsx"
XLSX_SHEET = "cardiovascular dataset"

#: The spreadsheet has a one-row banner above the real header.
XLSX_HEADER_ROW = 1

COL_SUBJECT = "subject_ID"
COL_LABEL = "Hypertension"
COL_SEX = "Sex(M/F)"
COL_AGE = "Age(year)"
COL_HEIGHT = "Height(cm)"
COL_WEIGHT = "Weight(kg)"
COL_SBP = "Systolic Blood Pressure(mmHg)"
COL_DBP = "Diastolic Blood Pressure(mmHg)"
COL_HR = "Heart Rate(b/m)"
COL_BMI = "BMI(kg/m^2)"
COL_DIABETES = "Diabetes"


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------

def _is_ppgbp_root(path: Path) -> bool:
    """True if ``path`` is a directory containing the PPG-BP layout."""
    try:
        return (path / SUBJECT_DIR_REL).is_dir() and (path / XLSX_REL).is_file()
    except OSError:
        return False


#: Locations probed in order when no explicit root is given.
_CANDIDATE_ROOTS = (
    "data/ppgbp",
    "./ppgbp",
    "/content/data/ppgbp",
    "/content/ppgbp",
    "/kaggle/working/data/ppgbp",
    "/kaggle/input/ppg-bp",
    "/kaggle/input/ppgbp",
    "/kaggle/input/ppg-bp-database",
)


def resolve_ppgbp_root(root: str | os.PathLike | None = None) -> Path:
    """Locate the extracted PPG-BP dataset.

    Search order: explicit ``root`` -> ``$PPGBP_ROOT`` -> common Colab/Kaggle
    /local locations -> a bounded recursive scan under ``/kaggle/input``.

    The returned directory is the one that *contains* ``Data File/``.
    """
    probes: list[Path] = []

    if root is not None:
        probes.append(Path(root))
    env = os.environ.get("PPGBP_ROOT")
    if env:
        probes.append(Path(env))
    probes.extend(Path(p) for p in _CANDIDATE_ROOTS)

    for cand in probes:
        cand = cand.expanduser()
        if _is_ppgbp_root(cand):
            return cand.resolve()
        # Tolerate being handed the "Data File" dir or the dir above it.
        if _is_ppgbp_root(cand.parent):
            return cand.parent.resolve()

    # Kaggle datasets get a slug we cannot predict; scan two levels deep.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for depth1 in sorted(kaggle_input.iterdir()):
            if _is_ppgbp_root(depth1):
                return depth1.resolve()
            if depth1.is_dir():
                for depth2 in sorted(depth1.iterdir()):
                    if _is_ppgbp_root(depth2):
                        return depth2.resolve()

    tried = "\n  ".join(str(p) for p in probes)
    raise FileNotFoundError(
        "Could not locate the PPG-BP dataset.\n"
        f"Looked for a directory containing '{SUBJECT_DIR_REL}' and "
        f"'{XLSX_REL}' in:\n  {tried}\n\n"
        "Fix by either:\n"
        "  python -m hyperppg.data.download --dest data/ppgbp\n"
        "or by setting PPGBP_ROOT=/path/to/extracted/dataset"
    )


def default_out_dir() -> Path:
    """A writable directory for checkpoints and reports."""
    for cand in (Path("/kaggle/working"), Path("/content")):
        if cand.is_dir() and os.access(cand, os.W_OK):
            return cand / "hyperppg_runs"
    return Path("runs")
