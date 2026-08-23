"""torch ``Dataset`` wrappers.

Augmentation happens on the 1-D waveform inside ``__getitem__``, *before*
rasterisation for the image models. That ordering matters: augmenting the
rendered image (rotations, flips) would produce waveforms that no PPG sensor
could ever emit, whereas noise, gain and warping applied to the signal stay
physiologically plausible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from hyperppg.data.augment import Transform
from hyperppg.data.render import waveform_to_image

__all__ = ["PPGImageDataset", "PPGSignalDataset", "PretrainWindowDataset", "make_sampler"]


class _RngMixin:
    """Per-worker RNG seeded from torch, so workers never share a stream."""

    _rng: np.random.Generator | None = None
    _seed_offset: int = 0

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            base = int(torch.initial_seed()) % (2**31)
            self._rng = np.random.default_rng(base + self._seed_offset)
        return self._rng


class PPGImageDataset(Dataset, _RngMixin):
    """Waveforms rendered to images -- the input the paper's CNNs expect.

    Parameters
    ----------
    signals
        ``(N, T)`` waveforms, already run through
        :func:`hyperppg.data.preprocess.paper_pipeline`.
    labels
        ``(N,)`` integer classes.
    augment
        Applied to the waveform before rendering. ``None`` for val/test.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        augment: Transform | None = None,
        height: int = 224,
        width: int = 224,
        line_width: int = 2,
        normalize: bool = True,
        seed_offset: int = 0,
    ):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.signals) != len(self.labels):
            raise ValueError(
                f"signals ({len(self.signals)}) and labels ({len(self.labels)}) differ"
            )
        self.augment = augment
        self.height = height
        self.width = width
        self.line_width = line_width
        self.normalize = normalize
        self._seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.signals[i]
        if self.augment is not None:
            x = self.augment(x.copy(), self.rng())
        img = waveform_to_image(
            x,
            height=self.height,
            width=self.width,
            line_width=self.line_width,
            channels=3,
            normalize=self.normalize,
        )
        return torch.from_numpy(img), int(self.labels[i])


class PPGSignalDataset(Dataset, _RngMixin):
    """Raw 1-D input for the hybrid encoder, with optional tabular covariates.

    ``renormalize`` re-applies z-scoring after augmentation; gain and smoothing
    transforms otherwise shift the scale the encoder was trained on.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        tabular: np.ndarray | None = None,
        augment: Transform | None = None,
        renormalize: bool = True,
        seed_offset: int = 0,
    ):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tabular = (
            np.asarray(tabular, dtype=np.float32) if tabular is not None else None
        )
        if self.tabular is not None and len(self.tabular) != len(self.labels):
            raise ValueError("tabular and labels length mismatch")
        self.augment = augment
        self.renormalize = renormalize
        self._seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.signals[i]
        if self.augment is not None:
            x = self.augment(x.copy(), self.rng())
        if self.renormalize:
            sd = float(x.std())
            x = (x - float(x.mean())) / max(sd, 1e-8)
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)

        if self.tabular is None:
            return x, int(self.labels[i])
        return x, torch.from_numpy(self.tabular[i]), int(self.labels[i])


class PretrainWindowDataset(Dataset, _RngMixin):
    """Unlabelled windows for masked-span self-supervision."""

    def __init__(
        self,
        windows: np.ndarray,
        augment: Transform | None = None,
        seed_offset: int = 0,
    ):
        self.windows = np.asarray(windows, dtype=np.float32)
        self.augment = augment
        self._seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        x = self.windows[i]
        if self.augment is not None:
            x = self.augment(x.copy(), self.rng())
            sd = float(x.std())
            x = (x - float(x.mean())) / max(sd, 1e-8)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)


def make_sampler(labels: np.ndarray, seed: int = 0):
    """A ``WeightedRandomSampler`` that balances the four classes per epoch.

    An alternative to class-weighted loss; using both at once over-corrects.
    """
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels).ravel()
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def class_weights(labels: np.ndarray, num_classes: int, device=None) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    labels = np.asarray(labels).ravel()
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32, device=device)
