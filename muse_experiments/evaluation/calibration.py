"""Constant-offset calibration for an already trained regressor.

The offset is chosen by recording-level W-bMAE. It is added after the
model prediction and the result is clipped to the competition range.
"""

import numpy as np

from muse_experiments.evaluation.metrics import weighted_bmae


def offset_grid(low=-120.0, high=40.0, step=1.0):
    """Inclusive one-second grid used to reproduce the -16 s result.

    ``linspace(-120, 40, 161)`` is the same set of offsets.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    count = int(round((high - low) / step)) + 1
    grid = low + step * np.arange(count, dtype=np.float64)
    if abs(grid[-1] - high) > 1e-8:
        raise ValueError("grid does not land on the requested high value")
    return grid


def apply_offset(prediction, offset, cap=600.0):
    """Add ``offset`` seconds, then clip to ``[0, cap]``."""
    shifted = np.asarray(prediction, dtype=np.float64) + float(offset)
    return np.clip(shifted, 0.0, float(cap))


def select_offset(frame, prediction, edges, weights, grid, cap=600.0):
    """Return the offset with the lowest recording-level W-bMAE.

    Ties keep the first grid point. The grid runs from negative to
    positive, so a tie keeps the more negative offset. ``frame`` must
    contain only the recordings that are allowed to choose the offset.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(prediction) != len(frame):
        raise ValueError("prediction length does not match the frame")
    best_offset = None
    best_score = None
    curve = []
    for offset in np.asarray(grid, dtype=np.float64):
        shifted = apply_offset(prediction, offset, cap)
        score = weighted_bmae(frame, shifted, edges, weights)
        curve.append((float(offset), float(score)))
        if best_score is None or score < best_score:
            best_score = float(score)
            best_offset = float(offset)
    if best_offset is None:
        raise RuntimeError("offset grid is empty")
    return best_offset, best_score, curve
