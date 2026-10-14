"""Causal temporal CNN for Muse sleep-onset windows.

Each 5-second window is encoded on its own. A dilated causal stack then
looks at the current window and earlier ones only. Normalization inside
the stack is per timestep, so a later window cannot change an earlier
hidden state.
"""

import torch
from torch import nn


class WindowEncoder(nn.Module):
    """1D CNN over one window of shape ``(C, samples)``."""

    def __init__(self, n_channels=4, width=32):
        super().__init__()
        if width % 4 != 0:
            raise ValueError("encoder width must be divisible by 4")
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel_size=9, stride=4, padding=4),
            nn.GroupNorm(4, width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(4, width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.width = width

    def forward(self, windows):
        return self.net(windows).squeeze(-1)


class CausalConv1d(nn.Module):
    """Convolution that pads only on the left."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, sequence):
        padded = nn.functional.pad(sequence, (self.left, 0))
        return self.conv(padded)


class TemporalBlock(nn.Module):
    """Residual dilated block. LayerNorm sees one timestep at a time."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv = CausalConv1d(channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, sequence):
        hidden = self.conv(sequence)
        hidden = hidden.transpose(1, 2)
        hidden = self.norm(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = self.drop(self.act(hidden))
        return sequence + hidden


class SleepOnsetNet(nn.Module):
    """Window encoder plus an optional causal temporal stack.

    ``sequence`` is ``(B, T, C, samples)``. ``mask`` is ``(B, T)`` with
    1 on real windows and 0 on left padding. Padded positions are zeroed
    after the encoder so a filler value cannot leak in.
    """

    def __init__(
        self,
        n_channels=4,
        width=32,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
        uv_scale=100.0,
        use_temporal=True,
        cap_seconds=600.0,
    ):
        super().__init__()
        self.encoder = WindowEncoder(n_channels, width)
        self.use_temporal = bool(use_temporal)
        blocks = []
        if self.use_temporal:
            for dilation in dilations:
                blocks.append(
                    TemporalBlock(width, kernel_size, int(dilation), dropout)
                )
        self.temporal = nn.Sequential(*blocks)
        self.head = nn.Linear(width, 1)
        self.uv_scale = float(uv_scale)
        self.cap_seconds = float(cap_seconds)

    def encode(self, sequence, mask):
        batch, steps, channels, samples = sequence.shape
        flat = sequence.reshape(batch * steps, channels, samples)
        flat = flat / self.uv_scale
        encoded = self.encoder(flat).reshape(batch, steps, -1)
        return encoded * mask.unsqueeze(-1).to(encoded.dtype)

    def forward_steps(self, sequence, mask):
        """One raw prediction per timestep, using only that step's past."""
        encoded = self.encode(sequence, mask)
        hidden = encoded.transpose(1, 2)
        if self.use_temporal:
            hidden = self.temporal(hidden)
        scores = self.head(hidden.transpose(1, 2)).squeeze(-1)
        return scores

    def forward(self, sequence, mask):
        return self.forward_steps(sequence, mask)[:, -1]

    def predict(self, sequence, mask):
        """Finite seconds in ``[0, cap]``, shape ``(B,)``."""
        with torch.no_grad():
            scores = self.forward(sequence, mask)
        return torch.clamp(scores, 0.0, self.cap_seconds)


def build_sleep_onset_net(cfg):
    """Build the network described by a team YAML config."""
    model_cfg = cfg["model"]
    return SleepOnsetNet(
        n_channels=len(cfg["data"]["channels"]),
        width=int(model_cfg["width"]),
        kernel_size=int(model_cfg["kernel_size"]),
        dilations=tuple(model_cfg["dilations"]),
        dropout=float(model_cfg["dropout"]),
        uv_scale=float(model_cfg["uv_scale"]),
        use_temporal=bool(model_cfg["use_temporal"]),
        cap_seconds=float(cfg["evaluation"]["cap_seconds"]),
    )
