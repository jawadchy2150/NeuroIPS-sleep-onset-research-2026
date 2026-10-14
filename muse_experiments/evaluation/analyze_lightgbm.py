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
from muse_experiments.io import publish_file, repo_path
from muse_experiments.training.lightgbm_run import split_subjects

RANGES = (
    ("After N2 / 0s", lambda frame: frame["target_s"] == 0),
    (
        "1-40s",
        lambda frame: (frame["target_s"] > 0) & (frame["target_s"] < 40),
    ),
    (
        "40-90s",
        lambda frame: (frame["target_s"] >= 40) & (frame["target_s"] < 90),
    ),
    (
        "90-300s",
        lambda frame: (frame["target_s"] >= 90) & (frame["target_s"] < 300),
    ),
    (
        "300-599s",
        lambda frame: (frame["target_s"] >= 300) & (frame["target_s"] < 600),
    ),
    ("600s", lambda frame: frame["target_s"] == 600),
)


def error_table(validation):
    rows = []
    for name, mask_fn in RANGES:
        subset = validation[mask_fn(validation)]
        if subset.empty:
            continue
        rows.append({
            "Range": name,
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

    print("Validation participants:", validation["subject"].nunique())
    print("Validation windows:", len(validation))
    print("Overall MAE:", round(validation["abs_error"].mean(), 2))
    table = error_table(validation)
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
