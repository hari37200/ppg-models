"""Waveform augmentations.

All transforms act on a single 1-D float32 segment and return one of the same
length, so they compose freely and can be applied on the fly inside a Dataset
(before rasterisation, for the image models).

The paper's own augmentation was "adding and removing noise to the PPG
signals"; :class:`AddGaussianNoise` and :class:`ExtraSmooth` cover that pair.
The rest are standard physiological-signal augmentations that matter a lot here
because the labelled set is only 657 segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

__all__ = [
    "Transform",
    "Compose",
    "RandomApply",
    "AddGaussianNoise",
    "ExtraSmooth",
    "AmplitudeScale",
    "BaselineWander",
    "TimeShift",
    "TimeWarp",
    "RandomCropResize",
    "Cutout",
    "PowerlineNoise",
    "default_train_augment",
    "mixup_batch",
]

Rng = np.random.Generator


class Transform:
    """Base class: subclasses implement ``__call__(x, rng) -> x``."""

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Compose(Transform):
    """Apply transforms in order."""

    transforms: list[Transform] = field(default_factory=list)

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        for t in self.transforms:
            x = t(x, rng)
        return x


@dataclass
class RandomApply(Transform):
    """Apply ``transform`` with probability ``p``."""

    transform: Transform
    p: float = 0.5

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        return self.transform(x, rng) if rng.random() < self.p else x


@dataclass
class AddGaussianNoise(Transform):
    """Additive white noise at a random SNR (dB)."""

    snr_db: tuple[float, float] = (15.0, 35.0)

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        power = float(np.mean(x.astype(np.float64) ** 2))
        if power <= 0:
            return x
        snr = rng.uniform(*self.snr_db)
        sigma = np.sqrt(power / (10.0 ** (snr / 10.0)))
        return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


@dataclass
class ExtraSmooth(Transform):
    """Additional moving average -- the "removing noise" half of the pair."""

    window: tuple[int, int] = (3, 25)

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        w = int(rng.integers(self.window[0], self.window[1] + 1))
        if w < 2:
            return x
        return uniform_filter1d(x, size=w, mode="nearest").astype(np.float32)


@dataclass
class AmplitudeScale(Transform):
    """Multiply by a random gain (perfusion / contact-pressure variation)."""

    scale: tuple[float, float] = (0.8, 1.25)

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        return (x * rng.uniform(*self.scale)).astype(np.float32)


@dataclass
class BaselineWander(Transform):
    """Add a slow sinusoid: respiration and probe drift."""

    amplitude: tuple[float, float] = (0.02, 0.15)
    freq_hz: tuple[float, float] = (0.05, 0.5)
    fs: float = 1000.0

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        t = np.arange(x.size, dtype=np.float32) / self.fs
        amp = rng.uniform(*self.amplitude) * (np.ptp(x) or 1.0)
        f = rng.uniform(*self.freq_hz)
        phase = rng.uniform(0, 2 * np.pi)
        return (x + amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


@dataclass
class PowerlineNoise(Transform):
    """Add mains hum. Only meaningful before decimation below 100 Hz."""

    amplitude: tuple[float, float] = (0.005, 0.05)
    freq_hz: float = 50.0
    fs: float = 1000.0

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        if self.freq_hz >= self.fs / 2:
            return x
        t = np.arange(x.size, dtype=np.float32) / self.fs
        amp = rng.uniform(*self.amplitude) * (np.ptp(x) or 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        return (x + amp * np.sin(2 * np.pi * self.freq_hz * t + phase)).astype(np.float32)


@dataclass
class TimeShift(Transform):
    """Circular shift -- the segment boundary is arbitrary anyway."""

    max_frac: float = 0.5

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        k = int(rng.integers(-int(self.max_frac * x.size), int(self.max_frac * x.size) + 1))
        return np.roll(x, k).astype(np.float32)


@dataclass
class TimeWarp(Transform):
    """Resample to a slightly different rate, then crop/pad back.

    Simulates heart-rate variation without changing pulse morphology.
    """

    rate: tuple[float, float] = (0.9, 1.1)

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        n = x.size
        r = rng.uniform(*self.rate)
        m = max(int(round(n * r)), 8)
        warped = sps.resample(x, m).astype(np.float32)
        if m >= n:
            start = int(rng.integers(0, m - n + 1))
            return warped[start : start + n]
        out = np.empty(n, dtype=np.float32)
        out[:m] = warped
        out[m:] = warped[-1]
        return out


@dataclass
class RandomCropResize(Transform):
    """Crop a random sub-window and stretch it back to full length."""

    min_frac: float = 0.75

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        n = x.size
        frac = rng.uniform(self.min_frac, 1.0)
        m = max(int(n * frac), 8)
        start = int(rng.integers(0, n - m + 1))
        return sps.resample(x[start : start + m], n).astype(np.float32)


@dataclass
class Cutout(Transform):
    """Blank out a short span (motion artefact / sensor dropout)."""

    max_frac: float = 0.1
    n_holes: int = 1

    def __call__(self, x: np.ndarray, rng: Rng) -> np.ndarray:
        out = x.copy()
        n = x.size
        for _ in range(self.n_holes):
            m = int(rng.integers(1, max(int(self.max_frac * n), 2)))
            start = int(rng.integers(0, n - m + 1))
            out[start : start + m] = float(np.mean(x))
        return out.astype(np.float32)


def default_train_augment(fs: float = 1000.0, strength: str = "medium") -> Compose:
    """A sensible augmentation stack for training.

    ``strength`` is one of ``"light"``, ``"medium"``, ``"strong"``.
    """
    if strength not in {"light", "medium", "strong"}:
        raise ValueError(f"unknown strength {strength!r}")

    p = {"light": 0.25, "medium": 0.4, "strong": 0.6}[strength]

    stack: list[Transform] = [
        RandomApply(AddGaussianNoise((20.0, 40.0) if strength == "light" else (12.0, 35.0)), p),
        RandomApply(ExtraSmooth(), p * 0.6),
        RandomApply(AmplitudeScale(), p),
        RandomApply(BaselineWander(fs=fs), p),
        RandomApply(TimeShift(), p),
        RandomApply(TimeWarp(), p * 0.8),
    ]
    if strength != "light":
        stack.append(RandomApply(RandomCropResize(), p * 0.6))
    if strength == "strong":
        stack.append(RandomApply(Cutout(), p * 0.5))
        stack.append(RandomApply(PowerlineNoise(fs=fs), p * 0.4))
    return Compose(stack)


def mixup_batch(
    x: np.ndarray,
    y_onehot: np.ndarray,
    alpha: float,
    rng: Rng,
) -> tuple[np.ndarray, np.ndarray]:
    """Convex-combine a batch with a shuffled copy of itself.

    Works for any input rank as long as the batch is axis 0, so the same call
    handles ``(B, T)`` signals and ``(B, C, H, W)`` images.
    """
    if alpha <= 0:
        return x, y_onehot
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(x.shape[0])
    shape = (-1,) + (1,) * (x.ndim - 1)
    lam_x = np.full(shape, lam, dtype=np.float32)
    return (
        (lam_x * x + (1 - lam_x) * x[perm]).astype(np.float32),
        (lam * y_onehot + (1 - lam) * y_onehot[perm]).astype(np.float32),
    )
