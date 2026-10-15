"""Codabench submission for the 120-second Muse TCN.

All inference Python lives in this file. ``tcn_weights.pt`` is the
state dict of the recording-level model. Each call to ``predict``
appends the given windows to a 24-window causal history. ``reset_state``
clears that history at a recording boundary.

The CNN embedding of a window is computed once and reused. Later
windows run only the causal temporal stack. The model uses
``meta["device"]`` when that device is available, and CPU otherwise.
"""

from pathlib import Path

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CHANNELS = ("TP9", "AF7", "AF8", "TP10")
SAMPLES = 640
SFREQ = 128.0
UV_SCALE = 100.0
CAP_S = 600.0
FALLBACK_S = 300.0
WIDTH = 32
HISTORY = 24
DILATIONS = (1, 2, 4, 8)
ARTIFACT = "tcn_weights.pt"


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


class MuseHistoryTCN(nn.Module):
    """Causal 24-window TCN. ``predict`` accepts ``(B, C, T)``."""

    def __init__(self, ch_names=None):
        super().__init__()
        self.encoder = WindowEncoder(len(CHANNELS), WIDTH)
        blocks = [
            TemporalBlock(WIDTH, 3, int(dilation), 0.1)
            for dilation in DILATIONS
        ]
        self.temporal = nn.Sequential(*blocks)
        self.head = nn.Linear(WIDTH, 1)
        self.uv_scale = UV_SCALE
        self.cap_seconds = CAP_S
        if ch_names is None:
            self.ch_names = None
            self._index = None
        else:
            self.ch_names = [str(name) for name in ch_names]
            self._index = [self.ch_names.index(name) for name in CHANNELS]
        self.reset_state()

    def reset_state(self):
        """Drop the rolling history at a recording boundary."""
        self._embeddings = []

    def encode(self, sequence, mask):
        batch, steps, channels, samples = sequence.shape
        flat = sequence.reshape(batch * steps, channels, samples)
        flat = flat / self.uv_scale
        encoded = self.encoder(flat).reshape(batch, steps, -1)
        return encoded * mask.unsqueeze(-1).to(encoded.dtype)

    def forward(self, sequence, mask):
        encoded = self.encode(sequence, mask)
        hidden = encoded.transpose(1, 2)
        hidden = self.temporal(hidden)
        scores = self.head(hidden.transpose(1, 2)).squeeze(-1)
        return scores[:, -1]

    def _prepare(self, batch):
        device = next(self.parameters()).device
        if isinstance(batch, torch.Tensor):
            data = batch.detach().to(device=device, dtype=torch.float32)
        else:
            data = torch.as_tensor(batch, dtype=torch.float32, device=device)
        if data.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(data.shape)}")
        if data.shape[-1] != SAMPLES:
            raise ValueError(
                f"expected {SAMPLES} samples, got {data.shape[-1]}"
            )
        if self._index is None:
            if data.shape[1] != len(CHANNELS):
                raise ValueError(
                    f"expected {len(CHANNELS)} channels, got {data.shape[1]}"
                )
            return data
        return data[:, self._index, :]

    def _temporal_score(self):
        """Score the newest embedding from cached history only."""
        current = self._embeddings[-1]
        sequence = current.new_zeros((1, WIDTH, HISTORY))
        kept = torch.stack(self._embeddings)
        sequence[0, :, -kept.shape[0]:] = kept.transpose(0, 1)
        hidden = self.temporal(sequence)
        latest = hidden.transpose(1, 2)[:, -1, :]
        return self.head(latest).reshape(())

    def predict(self, batch):
        """Finite seconds in ``[0, 600]``, shape ``(B,)``, on CPU."""
        data = self._prepare(batch)
        self.eval()
        with torch.inference_mode():
            embedded = self.encoder(data / self.uv_scale)
            scores = embedded.new_empty(embedded.shape[0])
            for row in range(embedded.shape[0]):
                self._embeddings.append(embedded[row].clone())
                if len(self._embeddings) > HISTORY:
                    del self._embeddings[:-HISTORY]
                scores[row] = self._temporal_score()
            scores = scores.clamp(0.0, self.cap_seconds)
            invalid = ~torch.isfinite(scores)
            if bool(invalid.any()):
                scores = torch.where(
                    invalid,
                    torch.full_like(scores, FALLBACK_S),
                    scores,
                )
        return scores.detach().cpu()


class Solver(CompetSolver):

    name = "Muse-TCN"

    def load_model(self, meta):
        path = Path(meta["submission_dir"]) / ARTIFACT
        if not path.is_file():
            raise FileNotFoundError(
                f"{ARTIFACT} is missing from {path.parent}"
            )
        _check_meta(meta)
        model = MuseHistoryTCN(ch_names=meta.get("ch_names"))
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        device = _select_device(meta.get("device"))
        if device.type == "cpu":
            # Tiny convolutions are slower when one op wakes every core.
            torch.set_num_threads(1)
        model.to(device)
        model.eval()
        model.reset_state()
        return model


def _select_device(requested):
    """Use the worker device, or CPU when it is missing."""
    name = "cpu" if requested is None else str(requested)
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _check_meta(meta):
    sfreq = meta.get("sfreq")
    if sfreq is not None and abs(float(sfreq) - SFREQ) > 1e-6:
        raise ValueError(f"submission expects {SFREQ} Hz, meta says {sfreq}")
    n_chans = meta.get("n_chans")
    if n_chans is not None and int(n_chans) != len(CHANNELS):
        raise ValueError(
            f"submission expects {len(CHANNELS)} channels, meta says {n_chans}"
        )
    n_times = meta.get("n_times")
    if n_times is not None and int(n_times) != SAMPLES:
        raise ValueError(
            f"submission expects {SAMPLES} samples, meta says {n_times}"
        )
