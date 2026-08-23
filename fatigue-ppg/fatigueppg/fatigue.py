"""The fatigue index -- Section 3.6 of the paper.

Per cycle: take the vertical height ``x`` from pulse onset to systolic peak,
put the zero at half of ``x``, and map zero -> systolic peak onto 0 -> 10, the
range of a BFI-Taiwan answer. The diastolic peak's position on that scale is
the cycle's fatigue index; a recording's index is the mean over its cycles.
"""
from __future__ import annotations

import numpy as np

from .config import FI_MAX, ZERO_FRAC

__all__ = ["cycle_fatigue_index", "dicrotic_ratio"]


def cycle_fatigue_index(x, onset, sys_idx, dia_idx, scale_max=FI_MAX,
                        zero_frac=ZERO_FRAC):
    """Position of the diastolic peak on the paper's 0-10 scale, one cycle.

    A consequence worth stating plainly, because it decides what the index can
    measure: with the zero at mid-height, **any diastolic peak below the middle
    of the pulse clips to 0**. The index is a half-wave-rectified measure of
    relative dicrotic-peak height. That suits the paper's cohort -- sixteen
    healthy 22-24-year-olds, whose reflected wave is strong -- and it is why
    the index floors on older or stiffer-arteried cohorts: on PPG-BP the median
    diastolic peak sits at 0.33 of the pulse height, so 82% of cycles read 0.

    ``zero_frac=0.0`` references the scale to the pulse onset instead and
    removes the floor. Keep the default 0.5 to reproduce the paper.
    """
    base, top = float(x[onset]), float(x[sys_idx])
    height = top - base                  # "vertical height of the systolic peak"
    if height <= 1e-9:
        return np.nan
    zero = base + zero_frac * height     # "half of x is the zero"
    span = top - zero
    if span <= 1e-9:
        return np.nan
    value = scale_max * (float(x[dia_idx]) - zero) / span
    return float(np.clip(value, 0.0, scale_max))


def dicrotic_ratio(x, onset, sys_idx, dia_idx):
    """Diastolic-peak height as a fraction of the pulse height (uncapped).

    The raw physiological quantity behind the index, useful for checking
    whether a cohort even reaches the index's zero point.
    """
    span = float(x[sys_idx]) - float(x[onset])
    if abs(span) < 1e-12:
        return np.nan
    return float((float(x[dia_idx]) - float(x[onset])) / span)
