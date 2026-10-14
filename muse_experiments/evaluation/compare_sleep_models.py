"""Compare saved sleep-onset models on the shared validation split.

Reads the corrected pre-N2 table and the existing checkpoints. It does
not train, and it does not modify those files. Outputs go to
``muse_experiments/results/analysis/``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import range_errors, weighted_bmae
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.preprocessing.windows import (
    SequenceDataset,
    ensure_window_cache,
    load_feature_table,
    recording_starts,
)
from muse_experiments.training.lightgbm_run import split_subjects

ANALYSIS = "muse_experiments/results/analysis"
BINS = (0.0, 40.0, 90.0, 300.0, 600.0)
WEIGHTS = (10.0, 5.0, 3.0, 1.0)
CAP = 600.0
SCORE_TOLERANCE = 0.02
MODEL_ORDER = (
    "constant_35s",
    "weighted_lightgbm",
    "cnn_5s",
    "tcn_120s",
)
LABELS = {
    "constant_35s": "Constant 35 s",
    "weighted_lightgbm": "Weighted LightGBM",
    "cnn_5s": "CNN (5 s)",
    "tcn_120s": "TCN (120 s)",
}
EXPECTED = {
    "constant_35s": {"mae": 336.37, "wbmae": 63.31},
    "weighted_lightgbm": {"mae": 195.29, "wbmae": 66.56},
    "cnn_5s": {"mae": 233.84, "wbmae": 76.56},
    "tcn_120s": {"mae": 243.52, "wbmae": 79.97},
}
EXPECTED_BIN_MAE = {
    "weighted_lightgbm": (53.10, 53.91, 82.85, 273.40),
    "cnn_5s": (75.83, 28.92, 95.09, 332.00),
    "tcn_120s": (80.66, 30.51, 89.61, 349.81),
}
COLORS = {
    "constant_35s": "#6b7280",
    "weighted_lightgbm": "#2563eb",
    "cnn_5s": "#d97706",
    "tcn_120s": "#059669",
}


def _edges():
    return list(BINS)


def _range_label(index):
    edges = _edges()
    low = edges[index]
    high = edges[index + 1]
    if index == len(edges) - 2:
        return f"[{low:g}, {high:g}]"
    return f"[{low:g}, {high:g})"


def _masks(targets):
    targets = np.asarray(targets, dtype=np.float64)
    edges = _edges()
    masks = []
    for index in range(len(edges) - 1):
        low = edges[index]
        high = edges[index + 1]
        if index == len(edges) - 2:
            masks.append((targets >= low) & (targets <= high))
        else:
            masks.append((targets >= low) & (targets < high))
    return masks


def _score(frame, prediction):
    mae = float(mean_absolute_error(frame["target_s"], prediction))
    score = weighted_bmae(frame, prediction, _edges(), WEIGHTS)
    return mae, score


def _check_scores(name, mae, score, bin_rows):
    expected = EXPECTED[name]
    mae_gap = abs(mae - expected["mae"])
    score_gap = abs(score - expected["wbmae"])
    if mae_gap > SCORE_TOLERANCE or score_gap > SCORE_TOLERANCE:
        raise RuntimeError(
            f"{name} did not reproduce the recorded scores: "
            f"MAE {mae:.4f} (expected {expected['mae']}), "
            f"W-bMAE {score:.4f} (expected {expected['wbmae']})"
        )
    recorded = EXPECTED_BIN_MAE.get(name)
    if recorded is None:
        return
    got = tuple(round(row["mae"], 2) for row in bin_rows)
    if got != tuple(round(value, 2) for value in recorded):
        gaps = [abs(a - b) for a, b in zip(got, recorded)]
        if max(gaps) > SCORE_TOLERANCE:
            raise RuntimeError(
                f"{name} bin MAE {got} != recorded {recorded}"
            )


def _lightgbm_prediction(frame):
    cfg = load_config("lightgbm_weighted")
    path = repo_path(cfg["inputs"]["model_file"])
    saved = joblib.load(path)
    features = list(saved["features"])
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"feature table is missing {missing}")
    raw = saved["model"].predict(frame[features])
    return np.clip(np.asarray(raw, dtype=np.float64), 0.0, CAP)


def _load_net(path):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    model = SleepOnsetNet(
        n_channels=len(blob["channels"]),
        width=int(blob["width"]),
        kernel_size=int(blob["kernel_size"]),
        dilations=tuple(blob["dilations"]),
        dropout=float(blob["dropout"]),
        uv_scale=float(blob["uv_scale"]),
        use_temporal=bool(blob["use_temporal"]),
        cap_seconds=float(blob["cap_seconds"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _neural_prediction(model, cache, frame, indices, history, device):
    dataset = SequenceDataset(cache, frame, indices, history)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    chunks = []
    targets = []
    with torch.no_grad():
        for sequence, mask, target, _subject, _session in loader:
            estimate = model.predict(
                sequence.to(device),
                mask.to(device),
            )
            chunks.append(estimate.detach().cpu().numpy())
            targets.append(target.numpy())
    prediction = np.concatenate(chunks).astype(np.float64)
    seen = np.concatenate(targets).astype(np.float64)
    return prediction, seen


def _cache_matches_features(cache, frame, indices, channels):
    """Cached microvolts must match the feature table's per-channel std."""
    checked = 0
    worst = 0.0
    for row in np.asarray(indices)[:: max(len(indices) // 20, 1)][:20]:
        window = np.asarray(cache[int(row)], dtype=np.float64)
        if window.shape != (len(channels), 640):
            raise ValueError(
                f"cache row {row} has shape {window.shape}"
            )
        for index, channel in enumerate(channels):
            cached = float(np.std(window[index]))
            stored = float(frame.iloc[int(row)][f"{channel}_std_uv"])
            worst = max(worst, abs(cached - stored))
        checked += 1
    if worst > 0.05:
        raise RuntimeError(
            "cached EEG std disagrees with the feature table "
            f"(max abs diff {worst:.4f} µV)"
        )
    sample = np.asarray(cache[int(indices[0])], dtype=np.float64)
    scale = float(np.median(np.std(sample, axis=1)))
    if not 0.5 <= scale <= 500:
        raise RuntimeError(
            f"cached window scale {scale:.4g} is not microvolts"
        )
    return {"rows_checked": checked, "max_std_abs_diff_uv": worst}


def _corr(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _within_recording_corr(frame, prediction, column):
    values = []
    grouped = frame.assign(_pred=prediction).groupby(
        ["subject", "session"], sort=False
    )
    for _key, part in grouped:
        values.append(_corr(part["_pred"], part[column]))
    usable = [item for item in values if np.isfinite(item)]
    if not usable:
        return float("nan")
    return float(np.mean(usable))


def _range_table(model, targets, prediction):
    rows = []
    for index, mask in enumerate(_masks(targets)):
        if not np.any(mask):
            continue
        actual = targets[mask]
        estimate = prediction[mask]
        rows.append({
            "model": model,
            "label": LABELS[model],
            "range": _range_label(index),
            "windows": int(mask.sum()),
            "mae": float(np.mean(np.abs(estimate - actual))),
            "mean_predicted_s": float(np.mean(estimate)),
            "mean_actual_s": float(np.mean(actual)),
            "bias_s": float(np.mean(estimate - actual)),
        })
    return rows


def _summary_row(model, frame, prediction, split_name):
    mae, score = _score(frame, prediction)
    return {
        "model": model,
        "label": LABELS[model],
        "split": split_name,
        "recordings": int(
            frame.groupby(["subject", "session"]).ngroups
        ),
        "subjects": int(frame["subject"].nunique()),
        "windows": int(len(frame)),
        "mae": mae,
        "wbmae": score,
        "mean_predicted_s": float(np.mean(prediction)),
        "mean_actual_s": float(frame["target_s"].mean()),
        "bias_s": float(np.mean(prediction - frame["target_s"])),
        "prediction_std_s": float(np.std(prediction)),
        "prediction_p10_s": float(np.percentile(prediction, 10)),
        "prediction_p50_s": float(np.percentile(prediction, 50)),
        "prediction_p90_s": float(np.percentile(prediction, 90)),
        "corr_with_target": _corr(prediction, frame["target_s"]),
        "within_recording_corr_target": _within_recording_corr(
            frame, prediction, "target_s"
        ),
        "within_recording_corr_elapsed": _within_recording_corr(
            frame, prediction, "window_end_s"
        ),
    }


def _subject_table(frame, predictions):
    rows = []
    keys = frame.groupby(["subject", "session"], sort=False).indices
    for (subject, session), positions in keys.items():
        part = frame.iloc[positions]
        row = {
            "subject": subject,
            "session": session,
            "windows": int(len(part)),
            "mean_actual_s": float(part["target_s"].mean()),
        }
        for model, prediction in predictions.items():
            estimate = np.asarray(prediction)[positions]
            mae, score = _score(part, estimate)
            row[f"{model}_mae"] = mae
            row[f"{model}_wbmae"] = score
            row[f"{model}_bias_s"] = float(
                np.mean(estimate - part["target_s"].to_numpy())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _slice_rows(model, frame, prediction, mask, slice_name):
    if int(np.sum(mask)) == 0:
        return None
    part = frame.loc[mask]
    estimate = np.asarray(prediction)[np.asarray(mask)]
    mae, score = _score(part, estimate)
    actual = part["target_s"].to_numpy()
    return {
        "model": model,
        "label": LABELS[model],
        "slice": slice_name,
        "windows": int(len(part)),
        "mae": mae,
        "wbmae": score,
        "mean_predicted_s": float(np.mean(estimate)),
        "mean_actual_s": float(np.mean(actual)),
        "bias_s": float(np.mean(estimate - actual)),
    }


def _position_masks(frame):
    order = frame.groupby(["subject", "session"], sort=False).cumcount()
    size = frame.groupby(
        ["subject", "session"], sort=False
    )["target_s"].transform("size")
    fraction = (order + 1) / size
    return fraction <= 0.5, fraction > 0.5


def _paired(frame, left_name, left, right_name, right):
    rows = []
    actual = frame["target_s"].to_numpy()
    left_abs = np.abs(left - actual)
    right_abs = np.abs(right - actual)
    groups = [( "all", np.ones(len(frame), dtype=bool))]
    groups.extend(
        (_range_label(index), mask)
        for index, mask in enumerate(_masks(actual))
    )
    for name, mask in groups:
        if not np.any(mask):
            continue
        better = right_abs[mask] < left_abs[mask]
        rows.append({
            "reference": left_name,
            "candidate": right_name,
            "range": name,
            "windows": int(mask.sum()),
            "candidate_lower_abs_error_fraction": float(np.mean(better)),
            "mae_delta_s": float(
                np.mean(right_abs[mask]) - np.mean(left_abs[mask])
            ),
        })
    return rows


def _write_csv(frame, name):
    desired = repo_path(f"{ANALYSIS}/{name}")

    def write_temp(path):
        frame.to_csv(path, index=False)

    return publish_file(desired, write_temp)


def _write_figure(figure, name):
    desired = repo_path(f"{ANALYSIS}/{name}")

    def write_temp(path):
        figure.savefig(path, dpi=140, bbox_inches="tight")

    return publish_file(desired, write_temp)


def _plot_scatter(frame, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = frame["target_s"].to_numpy()
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), sharex=True)
    for axis, model in zip(axes.ravel(), MODEL_ORDER):
        estimate = predictions[model]
        axis.scatter(
            actual,
            estimate,
            s=8,
            alpha=0.28,
            c=COLORS[model],
            linewidths=0,
            rasterized=True,
        )
        axis.plot([0, CAP], [0, CAP], color="#111827", lw=0.8, ls="--")
        mae, score = _score(frame, estimate)
        axis.set_title(
            f"{LABELS[model]}\nMAE {mae:.1f} s, W-bMAE {score:.1f} s",
            fontsize=10,
        )
        axis.set_xlim(0, CAP)
        axis.set_ylim(0, CAP)
        axis.set_xlabel("Actual seconds to first N2")
        axis.set_ylabel("Predicted seconds")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Validation predictions, 17 recordings, pre-N2 windows",
        fontsize=12,
    )
    figure.tight_layout()
    path = _write_figure(figure, "actual_vs_predicted.png")
    plt.close(figure)
    return path


def _time_bins(targets):
    width = 30.0
    edges = np.arange(0.0, CAP + width, width)
    edges[-1] = CAP
    index = np.digitize(targets, edges[1:-1], right=False)
    last = len(edges) - 2
    rows = []
    for bin_id in range(last + 1):
        if bin_id == last:
            mask = (targets >= edges[bin_id]) & (targets <= edges[-1])
        else:
            mask = (targets >= edges[bin_id]) & (targets < edges[bin_id + 1])
        if not np.any(mask):
            continue
        center = 0.5 * (edges[bin_id] + edges[bin_id + 1])
        rows.append((center, mask, int(mask.sum())))
    return rows


def _plot_error_curve(frame, predictions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = frame["target_s"].to_numpy()
    bins = _time_bins(actual)
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    for model in MODEL_ORDER:
        estimate = predictions[model]
        centers = []
        maes = []
        biases = []
        for center, mask, _count in bins:
            centers.append(center)
            maes.append(float(np.mean(np.abs(estimate[mask] - actual[mask]))))
            biases.append(float(np.mean(estimate[mask] - actual[mask])))
        axes[0].plot(
            centers,
            maes,
            marker="o",
            ms=3.5,
            lw=1.4,
            color=COLORS[model],
            label=LABELS[model],
        )
        axes[1].plot(
            centers,
            biases,
            marker="o",
            ms=3.5,
            lw=1.4,
            color=COLORS[model],
            label=LABELS[model],
        )
    for axis in axes:
        axis.axvspan(0, 90, color="#9ca3af", alpha=0.15, lw=0)
        axis.grid(True, axis="y", lw=0.4, color="#e5e7eb")
    axes[0].set_ylabel("Mean absolute error (seconds)")
    axes[1].set_ylabel("Bias, prediction minus actual (seconds)")
    axes[1].axhline(0, color="#111827", lw=0.7)
    axes[1].set_xlabel("Actual seconds until first N2 (30-second bin centers)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title(
        "Validation error across time-to-N2, 17 pre-N2 recordings"
    )
    figure.tight_layout()
    path = _write_figure(figure, "error_across_time_to_n2.png")
    plt.close(figure)
    return path


def _predict_all(frame, train_idx, val_idx):
    cfg = load_config("tcn_120s")
    channels = list(cfg["data"]["channels"])
    if channels != ["TP9", "AF7", "AF8", "TP10"]:
        raise RuntimeError(f"unexpected channel order: {channels}")
    root = repo_path(cfg["paths"]["muse_data_root"])
    cache = ensure_window_cache(frame, cfg, root)
    unit_check = _cache_matches_features(
        cache.array, frame, val_idx, channels
    )
    choice = resolve_device(cfg.get("device"))
    device = choice.device
    print("Torch device:", describe_device(choice))
    print(
        "Cache std check: "
        f"{unit_check['rows_checked']} rows, "
        f"max abs diff {unit_check['max_std_abs_diff_uv']:.5f} µV"
    )

    validation = frame.iloc[val_idx].reset_index(drop=True)
    training = frame.iloc[train_idx].reset_index(drop=True)
    expected_targets = validation["target_s"].to_numpy(dtype=np.float64)
    predictions = {
        "constant_35s": np.full(len(validation), 35.0),
        "weighted_lightgbm": _lightgbm_prediction(validation),
    }
    train_predictions = {
        "constant_35s": np.full(len(training), 35.0),
        "weighted_lightgbm": _lightgbm_prediction(training),
    }
    specs = (
        ("cnn_5s", "cnn_5s", 1, False),
        ("tcn_120s", "tcn_120s", 24, True),
    )
    for name, config_name, history, temporal in specs:
        path = repo_path(load_config(config_name)["outputs"]["model_file"])
        model, blob = _load_net(path)
        if int(blob["history_windows"]) != history:
            raise RuntimeError(
                f"{name} history is {blob['history_windows']}, "
                f"expected {history}"
            )
        if bool(blob["use_temporal"]) != temporal:
            raise RuntimeError(f"{name} temporal flag does not match")
        if list(blob["channels"]) != channels:
            raise RuntimeError(f"{name} channel order does not match")
        model = model.to(device)
        print(f"Scoring {name} on validation...")
        estimate, seen = _neural_prediction(
            model, cache, frame, val_idx, history, device
        )
        if not np.allclose(seen, expected_targets, rtol=0, atol=1e-5):
            raise RuntimeError(f"{name} targets do not match the table")
        predictions[name] = estimate
        print(f"Scoring {name} on training...")
        train_estimate, _train_seen = _neural_prediction(
            model, cache, frame, train_idx, history, device
        )
        train_predictions[name] = train_estimate
    lengths = {len(item) for item in predictions.values()}
    if lengths != {len(validation)}:
        raise RuntimeError(f"prediction lengths differ: {lengths}")
    return validation, training, predictions, train_predictions


def _reproduce(validation, predictions):
    for model in MODEL_ORDER:
        rows = range_errors(
            validation["target_s"], predictions[model], _edges()
        )
        mae, score = _score(validation, predictions[model])
        print(
            f"{LABELS[model]}: MAE {mae:.2f} s, W-bMAE {score:.2f} s"
        )
        _check_scores(model, mae, score, rows)


def compare():
    cfg = load_config("lightgbm_weighted")
    frame = load_feature_table(cfg)
    train_idx, val_idx = split_subjects(frame, cfg)
    validation, training, predictions, train_predictions = _predict_all(
        frame, train_idx, val_idx
    )
    if validation["has_official_test"].astype(int).sum() != 0:
        raise RuntimeError("validation includes an official test subject")
    if validation["subject"].nunique() != 17 or len(validation) != 2603:
        raise RuntimeError(
            "validation split is not the recorded 17 subjects / "
            f"2603 windows (got {validation['subject'].nunique()} / "
            f"{len(validation)})"
        )
    _reproduce(validation, predictions)

    summary_rows = []
    range_rows = []
    for model in MODEL_ORDER:
        summary_rows.append(
            _summary_row(model, validation, predictions[model], "validation")
        )
        summary_rows.append(
            _summary_row(
                model, training, train_predictions[model], "training"
            )
        )
        range_rows.extend(
            _range_table(
                model,
                validation["target_s"].to_numpy(),
                predictions[model],
            )
        )
    summary = pd.DataFrame(summary_rows)
    ranges = pd.DataFrame(range_rows)
    subjects = _subject_table(validation, predictions)

    early, late = _position_masks(validation)
    full_rows = frame
    starts = recording_starts(
        full_rows["subject"], full_rows["session"]
    )
    available = val_idx - starts[val_idx] + 1
    full_history = available >= 24
    slice_rows = []
    for model in MODEL_ORDER:
        estimate = predictions[model]
        for mask, name in (
            (early.to_numpy(), "early_half"),
            (late.to_numpy(), "late_half"),
            (full_history, "history_24_available"),
            (~full_history, "history_under_24"),
        ):
            row = _slice_rows(model, validation, estimate, mask, name)
            if row is not None:
                slice_rows.append(row)
        halves = (
            (early.to_numpy(), "early"),
            (late.to_numpy(), "late"),
        )
        for index, bin_mask in enumerate(
            _masks(validation["target_s"].to_numpy())
        ):
            for half, half_name in halves:
                row = _slice_rows(
                    model,
                    validation,
                    estimate,
                    bin_mask & half,
                    f"{half_name}|{_range_label(index)}",
                )
                if row is not None:
                    slice_rows.append(row)
    slices = pd.DataFrame(slice_rows)
    paired = pd.DataFrame(
        _paired(
            validation,
            "cnn_5s",
            predictions["cnn_5s"],
            "tcn_120s",
            predictions["tcn_120s"],
        )
    )

    detail = validation[
        [
            "subject",
            "session",
            "window_start_s",
            "window_end_s",
            "target_s",
        ]
    ].copy()
    detail["history_windows_available"] = available
    detail["recording_half"] = np.where(
        early.to_numpy(), "early", "late"
    )
    for model in MODEL_ORDER:
        detail[f"{model}_prediction_s"] = predictions[model]
        detail[f"{model}_abs_error_s"] = np.abs(
            predictions[model] - detail["target_s"].to_numpy()
        )
        detail[f"{model}_bias_s"] = (
            predictions[model] - detail["target_s"].to_numpy()
        )

    written = [
        _write_csv(summary, "four_model_comparison.csv"),
        _write_csv(ranges, "errors_by_target_range.csv"),
        _write_csv(subjects, "subject_performance.csv"),
        _write_csv(detail, "window_predictions.csv"),
        _write_csv(slices, "slice_errors.csv"),
        _write_csv(paired, "tcn_vs_cnn.csv"),
        _plot_scatter(validation, predictions),
        _plot_error_curve(validation, predictions),
    ]
    for path in written:
        print("Wrote", path)
    return summary, ranges, subjects, slices, paired


def main():
    compare()


if __name__ == "__main__":
    main()
