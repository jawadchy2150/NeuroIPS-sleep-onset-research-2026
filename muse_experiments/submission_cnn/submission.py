"""Codabench submission for the 5-second Muse CNN.

All inference Python lives in this file. ``cnn_weights.pt`` is the
state dict of the recording-level model, with the same encoder and the
fixed divisor of 100 used in training. The network scores only the
current 5-second window. Inference stays on CPU.
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
ARTIFACT = "cnn_weights.pt"


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


class MuseWindowCNN(nn.Module):
    """Current-window encoder. ``predict`` accepts ``(B, C, T)``."""

    def __init__(self, ch_names=None):
        super().__init__()
        self.encoder = WindowEncoder(len(CHANNELS), WIDTH)
        self.head = nn.Linear(WIDTH, 1)
        self.uv_scale = UV_SCALE
        self.cap_seconds = CAP_S
        if ch_names is None:
            self.ch_names = None
        else:
            self.ch_names = [str(name) for name in ch_names]

    def reset_state(self):
        """No rolling buffer. A recording reset changes nothing."""

    def encode(self, sequence, mask):
        batch, steps, channels, samples = sequence.shape
        flat = sequence.reshape(batch * steps, channels, samples)
        flat = flat / self.uv_scale
        encoded = self.encoder(flat).reshape(batch, steps, -1)
        return encoded * mask.unsqueeze(-1).to(encoded.dtype)

    def forward(self, sequence, mask):
        encoded = self.encode(sequence, mask)
        scores = self.head(encoded).squeeze(-1)
        return scores[:, -1]

    def _prepare(self, batch):
        if isinstance(batch, torch.Tensor):
            data = batch.detach().to(device="cpu", dtype=torch.float32)
        else:
            data = torch.as_tensor(batch, dtype=torch.float32, device="cpu")
        if data.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(data.shape)}")
        if data.shape[-1] != SAMPLES:
            raise ValueError(
                f"expected {SAMPLES} samples, got {data.shape[-1]}"
            )
        if self.ch_names is None:
            if data.shape[1] != len(CHANNELS):
                raise ValueError(
                    f"expected {len(CHANNELS)} channels, got {data.shape[1]}"
                )
            return data
        index = [self.ch_names.index(channel) for channel in CHANNELS]
        return data[:, index, :]

    def predict(self, batch):
        """Finite seconds in ``[0, 600]``, shape ``(B,)``, on CPU."""
        data = self._prepare(batch)
        self.eval()
        with torch.inference_mode():
            sequence = data.unsqueeze(1)
            mask = torch.ones(
                data.shape[0], 1, dtype=data.dtype, device=data.device
            )
            scores = self.forward(sequence, mask)
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

    name = "Muse-CNN"

    def load_model(self, meta):
        path = Path(meta["submission_dir"]) / ARTIFACT
        if not path.is_file():
            raise FileNotFoundError(
                f"{ARTIFACT} is missing from {path.parent}"
            )
        _check_meta(meta)
        model = MuseWindowCNN(ch_names=meta.get("ch_names"))
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to("cpu")
        model.eval()
        return model


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
