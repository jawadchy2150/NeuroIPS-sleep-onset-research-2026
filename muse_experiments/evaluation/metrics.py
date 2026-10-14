"""Local LightGBM checks. These are not the Codabench scorer."""

import numpy as np


def bin_ids(targets, edges):
    """Bin ids used by the weighted LightGBM script.

    Thresholds are the interior edges. With ``[0, 40, 90, 300, 600]`` the
    cuts are ``< 40``, ``< 90``, ``< 300``, and everything else.
    """
    thresholds = list(edges)[1:-1]
    conditions = [np.asarray(targets) < threshold for threshold in thresholds]
    choices = list(range(len(thresholds)))
    return np.select(conditions, choices, default=len(thresholds))


def bin_sample_weights(frame, edges, weights):
    """Per-window weights from the weighted training script."""
    train_frame = frame.copy()
    ids = bin_ids(train_frame["target_s"].to_numpy(), edges)
    train_frame["_bin"] = ids
    counts = train_frame.groupby(["subject", "session", "_bin"])[
        "_bin"
    ].transform("size").to_numpy()
    sample_weights = np.asarray(weights, dtype=float)[ids] / counts
    sample_weights = sample_weights / sample_weights.mean()
    return sample_weights


def weighted_bmae(frame, predictions, edges, weights):
    """Per-recording weighted bin MAE, then the mean across recordings."""
    bins = [
        (edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
    ]
    temp = frame.copy()
    temp["prediction"] = predictions
    recording_scores = []

    for _key, recording in temp.groupby(["subject", "session"]):
        target = recording["target_s"].to_numpy()
        estimate = recording["prediction"].to_numpy()
        errors = []
        used_weights = []
        for index, (low, high) in enumerate(bins):
            if index == len(bins) - 1:
                mask = (target >= low) & (target <= high)
            else:
                mask = (target >= low) & (target < high)
            if mask.any():
                errors.append(
                    np.mean(np.abs(target[mask] - estimate[mask]))
                )
                used_weights.append(weights[index])
        if errors:
            recording_scores.append(
                np.average(errors, weights=used_weights)
            )
    return float(np.mean(recording_scores))
