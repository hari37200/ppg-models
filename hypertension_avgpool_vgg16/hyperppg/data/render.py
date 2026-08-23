"""Rasterise a 1-D PPG waveform into the 2-D image the paper's CNNs consume.

The paper feeds "PPG images" -- plots of the waveform -- to AlexNet / ResNet-50
/ VGG-16. Going through matplotlib for every sample and every augmentation is
far too slow, so this module draws the polyline directly into a numpy array.

The result is a binary line drawing on a black background, optionally dilated
to a given stroke width and broadcast to 3 channels so ImageNet-pretrained
stems can be used unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["waveform_to_image", "batch_to_images", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _column_spans(values: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-column [top, bottom] row span covered by the waveform.

    ``values`` must already be scaled to [0, 1]. Returns two ``(width,)`` int
    arrays. Because a 2100-sample segment maps onto ~224 columns, each column
    receives ~9 samples and the min/max of those samples is an accurate span.
    """
    n = values.size
    cols = np.floor(np.linspace(0, width - 1e-9, n)).astype(np.intp)
    rows = (height - 1) - np.rint(np.clip(values, 0.0, 1.0) * (height - 1)).astype(np.intp)
    rows = np.clip(rows, 0, height - 1)

    top = np.full(width, height, dtype=np.intp)
    bottom = np.full(width, -1, dtype=np.intp)
    np.minimum.at(top, cols, rows)
    np.maximum.at(bottom, cols, rows)

    # Columns that received no sample (possible only if width > n) inherit
    # their left neighbour so the trace stays connected.
    empty = bottom < 0
    if empty.any():
        last_top, last_bottom = height // 2, height // 2
        for c in range(width):
            if empty[c]:
                top[c], bottom[c] = last_top, last_bottom
            else:
                last_top, last_bottom = top[c], bottom[c]

    # Bridge steep slopes: where two adjacent columns' spans do not overlap the
    # rasterised line would break, so stretch the current span to meet the
    # previous one.
    for c in range(1, width):
        if top[c] > bottom[c - 1]:
            top[c] = bottom[c - 1]
        elif bottom[c] < top[c - 1]:
            bottom[c] = top[c - 1]

    return top, bottom


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by shifting and OR-ing -- avoids a scipy dependency."""
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            # np.roll wraps; blank the wrapped-in border.
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            out |= shifted
    return out


def waveform_to_image(
    x: np.ndarray,
    height: int = 224,
    width: int = 224,
    line_width: int = 2,
    channels: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Draw one waveform as a ``(channels, height, width)`` float32 image.

    Parameters
    ----------
    x
        1-D signal. Rescaled to [0, 1] internally, so any input range works.
    line_width
        Stroke thickness in pixels (1 = hairline).
    normalize
        Apply ImageNet mean/std normalisation. Turn off to inspect the raw
        drawing.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 samples to draw a waveform")

    lo, hi = float(x.min()), float(x.max())
    v = (x - lo) / max(hi - lo, 1e-8)

    top, bottom = _column_spans(v, height, width)

    mask = np.zeros((height, width), dtype=bool)
    rr = np.arange(height)[:, None]
    mask |= (rr >= top[None, :]) & (rr <= bottom[None, :])

    mask = _dilate(mask, radius=max(line_width - 1, 0))

    img = mask.astype(np.float32)
    img = np.repeat(img[None, ...], channels, axis=0)

    if normalize and channels == 3:
        img = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return img


def batch_to_images(
    X: np.ndarray,
    height: int = 224,
    width: int = 224,
    line_width: int = 2,
    channels: int = 3,
    normalize: bool = True,
    progress: bool = False,
) -> np.ndarray:
    """Rasterise ``(N, T)`` into ``(N, channels, height, width)`` float32.

    Note the memory cost: 657 x 3 x 224 x 224 float32 is ~395 MB, which is fine,
    but rendering an augmented epoch up front is not -- augment on the fly
    instead (see :mod:`hyperppg.datasets`).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected (N, T), got {X.shape}")

    it = range(X.shape[0])
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, desc="rendering")
        except ImportError:
            pass

    out = np.empty((X.shape[0], channels, height, width), dtype=np.float32)
    for i in it:
        out[i] = waveform_to_image(
            X[i],
            height=height,
            width=width,
            line_width=line_width,
            channels=channels,
            normalize=normalize,
        )
    return out
