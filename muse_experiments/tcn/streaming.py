"""Rolling-buffer inference for the temporal sleep-onset network.

Codabench sends one window at a time. The buffer keeps at most the
configured history from the current recording. ``reset_state`` drops it.
"""

import numpy as np

from muse_experiments.preprocessing.windows import stack_history


def _as_numpy(batch):
    if hasattr(batch, "detach"):
        batch = batch.detach().cpu().numpy()
    return np.asarray(batch, dtype=np.float32)


class RecordingBuffer:
    """Apply a trained network to a stream of ``(B, C, samples)`` windows."""

    def __init__(self, model, history):
        self.model = model
        self.history = int(history)
        self.reset_state()

    def reset_state(self):
        self._windows = []

    def _score(self, sequence, mask):
        import torch

        device = next(self.model.parameters()).device
        seq = torch.as_tensor(sequence, device=device).unsqueeze(0)
        weights = torch.as_tensor(mask, device=device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.predict(seq, weights)
        value = prediction.detach().cpu().numpy().reshape(-1)
        if value.shape != (1,) or not np.isfinite(value).all():
            raise ValueError(f"expected one finite prediction, got {value}")
        return float(value[0])

    def predict(self, batch):
        data = _as_numpy(batch)
        if data.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {data.shape}")
        scores = np.empty(data.shape[0], dtype=np.float64)
        for index, window in enumerate(data):
            self._windows.append(np.asarray(window, dtype=np.float32))
            if len(self._windows) > self.history:
                self._windows = self._windows[-self.history:]
            sequence, mask = stack_history(self._windows, self.history)
            scores[index] = self._score(sequence, mask)
        return scores
