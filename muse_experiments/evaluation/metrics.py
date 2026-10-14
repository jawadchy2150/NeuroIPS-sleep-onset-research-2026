"""Warm-up W-bMAE, matching the upstream competition metric.

``binned_mae`` and ``group_scores`` follow
``benchmark_utils/metrics.py`` on the organizers' current main branch:
within each recording, non-empty bin MAEs are combined as
``sum(weight * MAE) / sum(weight)``, then those recording scores are
averaged. This checkout's copy of that file does not yet take weights.
"""

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


def binned_mae(
    y_true,
    y_pred,
    bin_edges=(0.0, 40.0, 90.0, 300.0, 600.0),
    bin_weights=None,
):
    """Per-bin MAE, then the weighted mean over non-empty bins."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1
    weights = (
        np.ones(n_bins)
        if bin_weights is None
        else np.asarray(bin_weights, dtype=np.float64)
    )
    if weights.shape != (n_bins,) or np.any(weights <= 0):
        raise ValueError(
            f"bin_weights must hold {n_bins} positive values, "
            f"one per bin; got {bin_weights!r}"
        )
    err = np.abs(y_pred - y_true)
    idx = np.digitize(y_true, edges[1:-1], right=False)
    in_range = (y_true >= edges[0]) & (y_true <= edges[-1])
    nonempty = [
        bin_id
        for bin_id in range(n_bins)
        if np.any(in_range & (idx == bin_id))
    ]
    if not nonempty:
        return float("nan")
    maes = np.array(
        [err[in_range & (idx == bin_id)].mean() for bin_id in nonempty]
    )
    used = weights[nonempty]
    return float((used * maes).sum() / used.sum())


def group_scores(metric, y_true, y_pred, groups):
    """One metric value per group, in sorted group-key order."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.array([])
    # String keys stay unicode. np.unique(..., axis=0) rejects object
    # arrays on this NumPy, which is what a pandas string column becomes.
    keys = np.asarray(groups).reshape(len(y_true), -1).astype(str)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return np.array([
        metric(y_true[inverse == group], y_pred[inverse == group])
        for group in range(inverse.max() + 1)
    ])


def weighted_bmae(frame, predictions, edges, weights):
    """W-bMAE per recording, then the mean across recordings."""
    y_true = frame["target_s"].to_numpy()
    groups = (
        frame["subject"].astype(str) + "|" + frame["session"].astype(str)
    ).to_numpy()

    def one_recording(target, estimate):
        return binned_mae(target, estimate, edges, weights)

    scores = group_scores(one_recording, y_true, predictions, groups)
    return float(np.mean(scores))


def range_errors(y_true, y_pred, edges):
    """MAE inside each official target bin. The last bin includes its cap."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    edges = list(edges)
    rows = []
    for index in range(len(edges) - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        last = index == len(edges) - 2
        if last:
            mask = (y_true >= low) & (y_true <= high)
            label = f"[{low:g}, {high:g}]"
        else:
            mask = (y_true >= low) & (y_true < high)
            label = f"[{low:g}, {high:g})"
        if not np.any(mask):
            continue
        error = np.abs(y_true[mask] - y_pred[mask])
        rows.append({
            "range": label,
            "windows": int(mask.sum()),
            "mae": float(error.mean()),
        })
    return rows
