"""The calibration model: what training produces and inference consumes.

There is no neural network here. The paper's "model" is Equation (9),

    subjective fatigue state = a + b * fatigue index

whose two coefficients were fitted by least squares on sixteen participants.
Training this package means fitting ``a`` and ``b`` on *your* labelled cohort;
inference means applying them to a new recording. A model file also records
which feature it was fitted on and how the index was computed, because a model
fitted on ``fi_onset`` cannot be applied to ``fi``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import (ALERT_THRESHOLD, DEFAULT_MODEL_PATH, PAPER_EQ9, PAPER_R_FI,
                     ZERO_FRAC)
from .stats import kfold_r, linreg

__all__ = ["FatigueModel", "load_model", "paper_model"]

SCHEMA_VERSION = 1


@dataclass
class FatigueModel:
    """A fitted Equation (7) plus the context needed to apply it safely."""

    a: float
    b: float
    feature: str = "fatigue_index"        # which column it was fitted on
    zero_frac: float = ZERO_FRAC          # how that column was computed
    threshold: float = ALERT_THRESHOLD    # rest-reminder cut-off, index units
    target: str = "subjective_fatigue_state"
    name: str = "unnamed"
    provenance: str = ""
    metrics: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    # -- use ---------------------------------------------------------------

    def predict(self, index):
        """Equation (9) applied to one index or an array of them."""
        return self.a + self.b * np.asarray(index, dtype=float)

    def alert(self, index) -> bool:
        """Whether the rest reminder fires (Section 4.4: index > 6)."""
        return bool(np.isfinite(index) and index > self.threshold)

    def equation(self) -> str:
        return f"{self.target} = {self.a:.4f} + {self.b:.4f} * {self.feature}"

    # -- persistence -------------------------------------------------------

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "FatigueModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema", 1) > SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written by a newer version of this package "
                f"(schema {data['schema']} > {SCHEMA_VERSION})")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(cls, index, subjective, feature="fatigue_index", zero_frac=ZERO_FRAC,
            name="fitted", provenance="", threshold=ALERT_THRESHOLD,
            folds=5, seed=0) -> "FatigueModel":
        """Least-squares fit of Equation (7), with an out-of-fold check.

        The in-sample ``r`` is what the paper reports. ``r_oof`` is what the
        model would score on participants it has not seen, and on a cohort this
        small the gap between them is the most informative number in the file.
        """
        fit = linreg(index, subjective)
        if not np.isfinite(fit["a"]):
            raise ValueError(
                f"cannot fit: only {fit['n']} usable (index, label) pairs")
        r_oof, mae_oof, n_oof = kfold_r(index, subjective, k=folds, seed=seed)
        metrics = {k: fit[k] for k in ("r", "p", "n", "stderr", "rmse", "mae")}
        metrics.update(r_oof=r_oof, mae_oof=mae_oof, n_oof=n_oof, folds=folds)
        return cls(a=fit["a"], b=fit["b"], feature=feature, zero_frac=zero_frac,
                   threshold=threshold, name=name, provenance=provenance,
                   metrics=metrics)

    def report(self) -> str:
        m = self.metrics
        lines = [f"model      {self.name}",
                 f"equation   {self.equation()}",
                 f"zero_frac  {self.zero_frac}   (0.5 = the paper's definition)",
                 f"threshold  {self.threshold} on the index"]
        if m:
            lines.append(
                f"in-sample  r = {m.get('r', float('nan')):.4f}  "
                f"p = {m.get('p', float('nan')):.3g}  n = {m.get('n', 0)}  "
                f"MAE = {m.get('mae', float('nan')):.3f}")
            if np.isfinite(m.get("r_oof", np.nan)):
                lines.append(
                    f"out-of-fold r = {m['r_oof']:.4f}  "
                    f"MAE = {m['mae_oof']:.3f}  ({m['folds']}-fold, "
                    f"n = {m['n_oof']})")
            else:
                lines.append("out-of-fold not computed (too few participants)")
        if self.provenance:
            lines.append(f"from       {self.provenance}")
        return "\n".join(lines)


def paper_model() -> FatigueModel:
    """Equation (9) exactly as published."""
    return FatigueModel(
        a=PAPER_EQ9[0], b=PAPER_EQ9[1], feature="fatigue_index",
        zero_frac=ZERO_FRAC, name="paper-eq9",
        provenance=("Chen et al., Mathematics 2023, 11, 3580, Equation (9); "
                    "fitted on 16 healthy adults aged 22-24 measured with a "
                    "COMGO device against the BFI-Taiwan form"),
        metrics=dict(r=PAPER_R_FI, n=16))


def load_model(path=None) -> FatigueModel:
    """Load a model file, falling back to the shipped paper coefficients."""
    if path is None:
        path = DEFAULT_MODEL_PATH
    path = Path(path)
    if not path.is_file():
        if Path(path) == Path(DEFAULT_MODEL_PATH):
            return paper_model()
        raise FileNotFoundError(f"no model at {path}")
    return FatigueModel.load(path)
