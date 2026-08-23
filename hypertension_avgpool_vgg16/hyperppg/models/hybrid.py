"""1-D PPG encoder and the hybrid classifier that improves on the paper.

Why not keep rasterising waveforms into 224x224 images? Because that route
throws away most of what a PPG carries. A 2100-sample segment becomes ~224
columns of ink -- a ~9x temporal decimation into a representation where a
130 M-parameter ImageNet stem has to *rediscover* that the picture is a
1-D time series. Operating on the signal directly is both cheaper and stronger.

Architecture
------------
``PPGEncoder1D``
    multi-scale conv stem (kernels 7/15/31 in parallel, so the first layer sees
    both the sharp systolic upstroke and the broad diastolic decay)
    -> 3 residual blocks with stride 2 (total 8x temporal downsampling)
    -> sinusoidal positional encoding + transformer encoder layers
    -> attention pooling to a single embedding

``PPGHybridClassifier``
    encoder + optional tabular-fusion branch (age / sex / BMI / HR) + head.

``MaskedPPGAutoencoder``
    the same encoder plus a light conv decoder, trained to inpaint masked
    spans. This is the self-supervised objective used to pretrain on the
    unlabelled wrist-PPG in PPG-DaLiA and FatigueSet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PPGEncoder1D",
    "PPGHybridClassifier",
    "MaskedPPGAutoencoder",
    "AttentionPool1d",
]


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class MultiScaleStem(nn.Module):
    """Parallel convolutions at several receptive fields, concatenated."""

    def __init__(self, in_ch: int = 1, out_ch: int = 64, kernels=(7, 15, 31)):
        super().__init__()
        if out_ch % len(kernels) != 0:
            raise ValueError(f"out_ch={out_ch} must be divisible by {len(kernels)}")
        per = out_ch // len(kernels)
        self.branches = nn.ModuleList(
            nn.Conv1d(in_ch, per, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        )
        self.norm = nn.BatchNorm1d(per * len(kernels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(self.norm(x))


class ResBlock1d(nn.Module):
    """Pre-activation residual block with optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.skip: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sin/cos encoding, computed on the fly for any length."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, D)``."""
        L = x.shape[1]
        if L > self.pe.shape[0]:
            raise ValueError(f"sequence length {L} exceeds max_len {self.pe.shape[0]}")
        return x + self.pe[:L].unsqueeze(0)


class AttentionPool1d(nn.Module):
    """Pool a token sequence with a single learned query."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, D)`` -> ``(B, D)``."""
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class PPGEncoder1D(nn.Module):
    """Conv + transformer encoder over a single-channel PPG segment.

    ``forward`` returns the pooled embedding; ``forward_tokens`` returns the
    ``(B, L', D)`` token sequence (used by the SSL decoder).
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 66,
        depths: tuple[int, ...] = (1, 1, 1),
        channels: tuple[int, ...] = (96, 144, 192),
        n_heads: int = 4,
        n_transformer_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(depths) != len(channels):
            raise ValueError("depths and channels must have equal length")

        self.stem = MultiScaleStem(in_ch, base_ch)

        blocks: list[nn.Module] = []
        prev = base_ch
        for n_rep, ch in zip(depths, channels):
            for i in range(n_rep):
                blocks.append(
                    ResBlock1d(prev, ch, stride=2 if i == 0 else 1, dropout=dropout)
                )
                prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.downsample_factor = 2 ** len(channels)

        d_model = prev
        self.d_model = d_model
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only emits a
        # warning; disable it explicitly since we never pass padding masks.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_transformer_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool1d(d_model)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 1, T)`` -> ``(B, L', d_model)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        h = self.stem(x)
        h = self.blocks(h)
        h = h.transpose(1, 2)  # (B, L', C)
        h = self.pos(h)
        h = self.transformer(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_tokens(x))

    @property
    def out_dim(self) -> int:
        return self.d_model


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

class PPGHybridClassifier(nn.Module):
    """Encoder + optional tabular fusion + classification head.

    Setting ``n_tabular > 0`` fuses clinical covariates. Be explicit about this
    when reporting: age and BMI are strongly predictive of hypertension stage on
    their own, so a fused model is no longer a pure-PPG result. Both variants
    are trained and reported separately by ``train_hybrid.py``.
    """

    def __init__(
        self,
        num_classes: int = 4,
        n_tabular: int = 0,
        encoder: PPGEncoder1D | None = None,
        tabular_dim: int = 32,
        head_dropout: float = 0.3,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else PPGEncoder1D(**encoder_kwargs)
        self.n_tabular = int(n_tabular)

        fused = self.encoder.out_dim
        if self.n_tabular > 0:
            self.tabular = nn.Sequential(
                nn.Linear(self.n_tabular, tabular_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(tabular_dim, tabular_dim),
                nn.GELU(),
            )
            fused += tabular_dim
        else:
            self.tabular = None

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(fused, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x)
        if self.tabular is not None:
            if tab is None:
                raise ValueError("model was built with n_tabular > 0 but tab is None")
            z = torch.cat([z, self.tabular(tab)], dim=1)
        return self.head(z)


# --------------------------------------------------------------------------
# Self-supervised pretraining
# --------------------------------------------------------------------------

class MaskedPPGAutoencoder(nn.Module):
    """Masked-span inpainting over PPG windows.

    Random contiguous spans of the input are overwritten with a learned mask
    embedding; the decoder must reconstruct the original samples. Loss is
    computed on masked positions only, so the model cannot win by copying.
    """

    def __init__(
        self,
        encoder: PPGEncoder1D | None = None,
        decoder_ch: int = 96,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else PPGEncoder1D(**encoder_kwargs)
        d = self.encoder.out_dim
        self.mask_token = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.Sequential(
            nn.Conv1d(d, decoder_ch, 5, padding=2),
            nn.BatchNorm1d(decoder_ch),
            nn.GELU(),
            nn.Conv1d(decoder_ch, decoder_ch, 5, padding=2),
            nn.BatchNorm1d(decoder_ch),
            nn.GELU(),
            nn.Conv1d(decoder_ch, 1, 1),
        )

    @staticmethod
    def make_span_mask(
        shape: tuple[int, int],
        mask_ratio: float,
        span: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Boolean ``(B, T)`` mask covering ~``mask_ratio`` of each row in spans.

        Spans are drawn independently and may overlap, so ``mask_ratio * T /
        span`` would systematically undershoot. The number of spans is instead
        solved from the expected coverage of ``n`` independent spans,
        ``1 - (1 - span/T)**n``, which lands on the requested ratio.
        """
        B, T = shape
        span = max(1, min(span, T))
        mask_ratio = float(min(max(mask_ratio, 0.0), 0.95))
        p = span / T
        if p >= 1.0:
            n_spans = 1
        else:
            n_spans = max(1, int(round(math.log1p(-mask_ratio) / math.log1p(-p))))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts = torch.randint(
            0, max(T - span, 1), (B, n_spans), device=device, generator=generator
        )
        offsets = torch.arange(span, device=device)
        idx = (starts.unsqueeze(-1) + offsets).clamp_(max=T - 1)  # (B, n_spans, span)
        mask.scatter_(1, idx.reshape(B, -1), True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
        span: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(reconstruction, mask, loss)``.

        ``x``: ``(B, 1, T)``. ``mask``: ``(B, T)`` boolean, True where hidden.
        """
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(f"expected (B, 1, T), got {tuple(x.shape)}")
        B, _, T = x.shape

        mask = self.make_span_mask((B, T), mask_ratio, span, x.device)
        corrupted = torch.where(mask.unsqueeze(1), self.mask_token.expand_as(x), x)

        tokens = self.encoder.forward_tokens(corrupted)  # (B, L', D)
        h = tokens.transpose(1, 2)  # (B, D, L')
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        recon = self.decoder(h)  # (B, 1, T)

        target = x
        diff = (recon - target) ** 2
        m = mask.unsqueeze(1).float()
        denom = m.sum().clamp_min(1.0)
        loss = (diff * m).sum() / denom
        return recon, mask, loss


def transfer_encoder_weights(
    ssl_checkpoint: str,
    classifier: PPGHybridClassifier,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict:
    """Load encoder weights from an SSL checkpoint into a classifier.

    Returns a report dict with the number of tensors loaded plus any
    missing/unexpected keys, so a silent no-op transfer is impossible to miss.
    """
    ckpt = torch.load(ssl_checkpoint, map_location=map_location, weights_only=False)
    state = ckpt.get("encoder_state", ckpt.get("state_dict", ckpt))

    # Strip a leading "encoder." if the whole autoencoder was saved.
    cleaned = {}
    for k, v in state.items():
        key = k[len("encoder.") :] if k.startswith("encoder.") else k
        cleaned[key] = v

    result = classifier.encoder.load_state_dict(cleaned, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    n_loaded = len(cleaned) - len(unexpected)
    if n_loaded == 0:
        raise RuntimeError(
            f"transferred 0 tensors from {ssl_checkpoint}; key names do not match"
        )
    return {
        "n_loaded": n_loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
