"""Train one LightGBM config and write a new checkpoint."""

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import (
    bin_sample_weights,
    binned_mae,
    range_errors,
    weighted_bmae,
)
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.booster import write_booster_json
from muse_experiments.models.lightgbm import build_regressor
from muse_experiments.preprocessing.features import METADATA_COLUMNS
from muse_experiments.splits import (
    assert_disjoint,
    official_muse_split,
    participant_split,
)


def feature_frame(df):
    features = [
        column for column in df.columns if column not in METADATA_COLUMNS
    ]
    return df[features], features


def split_subjects(df, cfg):
    column = cfg["split"]["group_column"]
    strategy = cfg["split"]["strategy"]
    if strategy == "official_muse_valid":
        if "has_official_test" not in df.columns:
            raise ValueError(
                "official_muse_valid needs a has_official_test column. "
                "Use the corrected pre-N2 feature table, not the earlier "
                "CSV that kept windows after N2."
            )
        train_idx, val_idx = official_muse_split(
            df[column].to_numpy(),
            df["has_official_test"].to_numpy(),
            val_size=cfg["split"]["val_size"],
            seed=cfg["split"]["seed"],
        )
    elif strategy == "group_shuffle_by_subject":
        train_idx, val_idx = participant_split(
            df[column],
            val_size=cfg["split"]["val_size"],
            seed=cfg["split"]["seed"],
        )
    else:
        raise ValueError(f"unknown split strategy: {strategy}")
    assert_disjoint(df.iloc[train_idx][column], df.iloc[val_idx][column])
    return train_idx, val_idx


def fit_lightgbm(df, cfg):
    """Fit on a subject-disjoint training split and score validation."""
    features_frame, features = feature_frame(df)
    target = df["target_s"]
    train_idx, val_idx = split_subjects(df, cfg)
    x_train = features_frame.iloc[train_idx]
    x_val = features_frame.iloc[val_idx]
    y_train = target.iloc[train_idx]
    y_val = target.iloc[val_idx]
    column = cfg["split"]["group_column"]
    print(
        "Training participants:",
        df.iloc[train_idx][column].nunique(),
    )
    print(
        "Validation participants:",
        df.iloc[val_idx][column].nunique(),
    )
    print("Training windows:", len(train_idx))
    print("Validation windows:", len(val_idx))
    print("Features:", len(features))

    choice = resolve_device(cfg.get("device"))
    print("Torch device:", describe_device(choice))
    print("LightGBM training runs on CPU.")
    model = build_regressor(cfg)
    sample_weight = None
    if cfg["training"]["sample_weight"]:
        sample_weight = bin_sample_weights(
            df.iloc[train_idx],
            cfg["evaluation"]["bins"],
            cfg["evaluation"]["bin_weights"],
        )
    started = time.perf_counter()
    if sample_weight is None:
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    elapsed = time.perf_counter() - started
    print(f"Training duration: {elapsed:.1f} seconds")

    cap = cfg["evaluation"]["cap_seconds"]
    prediction = np.clip(model.predict(x_val), 0, cap)
    bins = cfg["evaluation"]["bins"]
    weights = cfg["evaluation"]["bin_weights"]
    print(
        "\nValidation MAE:",
        round(mean_absolute_error(y_val, prediction), 2),
        "seconds",
    )
    val_frame = df.iloc[val_idx].copy()
    score = weighted_bmae(val_frame, prediction, bins, weights)
    print("W-bMAE (mean over recordings):", round(score, 2), "seconds")
    pooled = binned_mae(y_val, prediction, bins, None)
    print("Pooled unweighted bMAE:", round(pooled, 2), "seconds")
    for row in range_errors(y_val, prediction, bins):
        print(
            f"  {row['range']}: MAE {row['mae']:.2f} s "
            f"({row['windows']} windows)"
        )
    median_value = float(np.median(y_train))
    for label, constant in (
        (f"training-median {median_value:.1f}s", median_value),
        ("constant 35s", 35.0),
    ):
        baseline = np.full(len(y_val), constant)
        baseline_score = weighted_bmae(
            val_frame, baseline, bins, weights
        )
        print(
            f"Constant baseline W-bMAE ({label}):",
            round(baseline_score, 2),
            "seconds",
        )
    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10))
    return {
        "model": model,
        "features": features,
        "channels": list(cfg["data"]["channels"]),
        "bands": cfg["data"]["bands"],
        "config": cfg.get("name"),
        "split_seed": cfg["split"]["seed"],
        "model_seed": cfg["model"]["random_state"],
        "sampling_hz": cfg["data"]["sampling_hz"],
        "window_samples": cfg["data"]["window_samples"],
        "device": choice.name,
        "device_requested": choice.requested,
    }


def save_model(payload, desired):
    def write_temp(temporary):
        joblib.dump(payload, temporary)

    path = publish_file(desired, write_temp)
    json_path = write_booster_json(payload, path)
    print("\nModel saved successfully!")
    print("Saved model:", path)
    if json_path is not None:
        print("Saved portable booster:", json_path)
    return path


def train_from_config(cfg):
    features_csv = repo_path(cfg["inputs"]["features_csv"])
    if not features_csv.is_file():
        raise FileNotFoundError(f"feature table not found: {features_csv}")
    frame = pd.read_csv(features_csv)
    payload = fit_lightgbm(frame, cfg)
    return save_model(payload, repo_path(cfg["outputs"]["model_file"]))


def main(default_config):
    args = sys.argv[1:]
    name = args[0] if args else default_config
    train_from_config(load_config(name))
