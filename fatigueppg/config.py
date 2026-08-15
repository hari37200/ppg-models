"""Constants from the paper, and where things live on disk.

Every number here is traceable to a section of

    Chen, Y.-X. et al. "Fatigue Estimation Using Peak Features from PPG
    Signals." Mathematics 2023, 11, 3580. doi:10.3390/math11163580
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Acquisition (Section 3.2.2)
# --------------------------------------------------------------------------

#: Sampling rate of the COMGO device used in the paper.
FS_PAPER = 200.0

#: Length of one measurement, in seconds.
DURATION_PAPER = 120.0

#: Empatica E4 BVP rate (PPG-DaLiA and FatigueSet both).
E4_FS = 64.0

# --------------------------------------------------------------------------
# Preprocessing (Section 3.3)
# --------------------------------------------------------------------------

BAND_LOW, BAND_HIGH, BAND_ORDER = 0.5, 8.0, 4

# --------------------------------------------------------------------------
# Peak detection (Section 3.4)
# --------------------------------------------------------------------------

#: Plausible beat period in seconds; step 2E of Section 3.4.1 rejects anything
#: outside this and widens the search block.
RR_MIN, RR_MAX = 0.3, 1.5

#: Initial search block and its increment (Section 3.4.1, steps 2 and 2E).
CYCLE0, CYCLE_STEP = 10, 5

#: How far past the systolic peak the dicrotic wave is looked for, as a
#: fraction of the distance to the next pulse onset (Section 3.4.3).
SEARCH_FRAC = 0.5

# --------------------------------------------------------------------------
# HRV (Section 3.5)
# --------------------------------------------------------------------------

VLF_BAND = (0.003, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

#: The paper resamples the R-R series to 250 Hz before the FFT.
HRV_FS_INTERP = 250.0

#: Below this much data the frequency-domain indices are not reported.
HRV_MIN_SECONDS = 60.0

# --------------------------------------------------------------------------
# Fatigue index (Section 3.6)
# --------------------------------------------------------------------------

#: The index is mapped onto the 0-10 range of a BFI-Taiwan answer.
FI_MAX = 10.0

#: "Half of x is the zero." 0.5 reproduces the paper; 0.0 references the scale
#: to the pulse onset and removes the index's floor. See docs in fatigue.py.
ZERO_FRAC = 0.5

# --------------------------------------------------------------------------
# Regression and the evaluation system (Sections 3.7, 3.9, 4.3, 4.4)
# --------------------------------------------------------------------------

#: Equation (9): subjective fatigue state = 3.1 + 0.6 * fatigue index.
PAPER_EQ9 = (3.1, 0.6)

#: Reported correlations of the revised subjective state against each index.
PAPER_R_FI = 0.907
PAPER_R_NHF = 0.14875

#: The system reminds the user to rest above this index (Section 4.4).
ALERT_THRESHOLD = 6.0

#: BFI-Taiwan items the paper averaged into the "revised subjective state"
#: (Q2 r = 0.8743, Q3 r = 0.5328).
BFI_ITEMS = (2, 3)

BFI_QUESTIONS = (
    "Q1 current level of fatigue",
    "Q2 general fatigue, past 24 h",
    "Q3 most exhaustion, past 24 h",
    "Q4 effect on general activity",
    "Q5 effect on mood",
    "Q6 effect on walking ability",
    "Q7 effect on daily work",
    "Q8 effect on social interaction",
    "Q9 effect on enjoyment of life",
)

# --------------------------------------------------------------------------
# Signal quality (not in the paper -- see quality.py for why it is here)
# --------------------------------------------------------------------------

#: Windows below this template-correlation are dropped from batch analyses.
#: A clean pulse train scores > 0.97, white noise < 0.5.
MIN_SQI = 0.70

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
MODELS_DIR = REPO_ROOT / "models"
EXAMPLES_DIR = REPO_ROOT / "examples"

#: Shipped default: the paper's own coefficients.
DEFAULT_MODEL_PATH = MODELS_DIR / "paper_eq9.json"


def default_data_dir() -> Path:
    """Writable place for downloaded corpora."""
    env = os.environ.get("FATIGUEPPG_DATA")
    if env:
        return Path(env)
    for cand in (Path("/kaggle/working"), Path("/content")):
        if cand.is_dir() and os.access(cand, os.W_OK):
            return cand / "data"
    return REPO_ROOT / "data"


# --------------------------------------------------------------------------
# Public corpora
# --------------------------------------------------------------------------

FIGSHARE_PPGBP = "https://ndownloader.figshare.com/files/9441097"

PPGBP_MARKER = "Data File/0_subject"
DALIA_MARKER = "S*/S*_E4.zip"
FATIGUESET_MARKER = "*/*/wrist_bvp.csv"

PPGBP_N_SEGMENTS, PPGBP_N_SUBJECTS, PPGBP_FS = 657, 219, 1000.0
