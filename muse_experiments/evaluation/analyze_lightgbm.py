"""Error table for a saved LightGBM model on its validation subjects."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd

from muse_experiments.config import load_config
from muse_experiments.evaluation.metrics import binned_mae, weighted_bmae
from muse_experiments.io import publish_file, repo_path
from muse_experiments.training.lightgbm_run import split_subjects


def _bin_mask(target, low, high, last):
    if last:
        return (target >= low) & (target <= high)
    return (target >= low) & (target < high)


def error_table(validation, edges):
    """MAE, mean prediction, and bias inside each official target bin."""
    rows = []
    target = validation["target_s"]
    for index in range(len(edges) - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        last = index == len(edges) - 2
        label = f"[{low:g}, {high:g}]" if last else f"[{low:g}, {high:g})"
        subset = validation[_bin_mask(target, low, high, last)]
        if subset.empty:
            continue
        rows.append({
            "Range": label,
            "Windows": len(subset),
            "MAE": round(subset["abs_error"].mean(), 2),
            "Avg prediction": round(subset["prediction"].mean(), 2),
            "Bias": round(subset["signed_error"].mean(), 2),
        })
    return pd.DataFrame(rows)


def analyze(cfg):
    features_csv = repo_path(cfg["inputs"]["features_csv"])
    model_file = repo_path(cfg["inputs"]["model_file"])
    if not features_csv.is_file():
        raise FileNotFoundError(f"feature table not found: {features_csv}")
    if not model_file.is_file():
        raise FileNotFoundError(f"model not found: {model_file}")

    frame = pd.read_csv(features_csv)
    saved = joblib.load(model_file)
    model = saved["model"]
    features = saved["features"]
    _train_idx, val_idx = split_subjects(frame, cfg)
    validation = frame.iloc[val_idx].copy()
    cap = cfg["evaluation"]["cap_seconds"]
    validation["prediction"] = np.clip(
        model.predict(validation[features]), 0, cap
    )
    validation["abs_error"] = (
        validation["target_s"] - validation["prediction"]
    ).abs()
    validation["signed_error"] = (
        validation["prediction"] - validation["target_s"]
    )

    bins = cfg["evaluation"]["bins"]
    bin_weights = cfg["evaluation"]["bin_weights"]
    print("Validation participants:", validation["subject"].nunique())
    print("Validation windows:", len(validation))
    print("Overall MAE:", round(validation["abs_error"].mean(), 2))
    score = weighted_bmae(
        validation,
        validation["prediction"].to_numpy(),
        bins,
        bin_weights,
    )
    print("W-bMAE (mean over recordings):", round(score, 2))
    pooled = binned_mae(
        validation["target_s"], validation["prediction"], bins, None
    )
    print("Pooled unweighted bMAE:", round(pooled, 2))
    table = error_table(validation, bins)
    print("\nError analysis by time range:")
    print(table.to_string(index=False))
    subject_errors = (
        validation.groupby("subject")["abs_error"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n5 participants with highest MAE:")
    print(subject_errors.head(5).round(2))
    print("\n5 participants with lowest MAE:")
    print(subject_errors.tail(5).round(2))
    return table


def write_table(table, desired):
    def write_temp(temporary):
        table.to_csv(temporary, index=False)

    path = publish_file(desired, write_temp)
    print("\nAnalysis saved successfully!")
    print("Saved analysis:", path)
    return path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    name = args[0] if args else "lightgbm_100"
    cfg = load_config(name)
    table = analyze(cfg)
    write_table(table, repo_path(cfg["outputs"]["analysis_csv"]))


if __name__ == "__main__":
    main()
