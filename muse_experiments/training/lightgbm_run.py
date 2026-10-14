"""Train one LightGBM config and write a new checkpoint."""

import sys
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
    weighted_bmae,
)
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.lightgbm import build_regressor
from muse_experiments.preprocessing.features import METADATA_COLUMNS
from muse_experiments.splits import assert_disjoint, participant_split


def feature_frame(df):
    features = [
        column for column in df.columns if column not in METADATA_COLUMNS
    ]
    return df[features], features


def split_subjects(df, cfg):
    column = cfg["split"]["group_column"]
    train_idx, val_idx = participant_split(
        df[column],
        val_size=cfg["split"]["val_size"],
        seed=cfg["split"]["seed"],
    )
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
    print("Features:", len(features))

    choice = resolve_device(cfg.get("device"))
    print("Torch device:", describe_device(choice))
    print("LightGBM training runs on CPU.")
    model = build_regressor(cfg)
    if cfg["training"]["sample_weight"]:
        weights = bin_sample_weights(
            df.iloc[train_idx],
            cfg["evaluation"]["bins"],
            cfg["evaluation"]["bin_weights"],
        )
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)

    cap = cfg["evaluation"]["cap_seconds"]
    prediction = np.clip(model.predict(x_val), 0, cap)
    print(
        "\nValidation MAE:",
        round(mean_absolute_error(y_val, prediction), 2),
        "seconds",
    )
    val_frame = df.iloc[val_idx].copy()
    score = weighted_bmae(
        val_frame,
        prediction,
        cfg["evaluation"]["bins"],
        cfg["evaluation"]["bin_weights"],
    )
    print("Approximate W-bMAE:", round(score, 2), "seconds")
    baseline = np.full(len(y_val), np.median(y_train))
    baseline_score = weighted_bmae(
        val_frame,
        baseline,
        cfg["evaluation"]["bins"],
        cfg["evaluation"]["bin_weights"],
    )
    print(
        "Constant baseline W-bMAE:",
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
        "config": cfg.get("name"),
        "device": choice.name,
        "device_requested": choice.requested,
    }


def save_model(payload, desired):
    def write_temp(temporary):
        joblib.dump(payload, temporary)

    path = publish_file(desired, write_temp)
    print("\nModel saved successfully!")
    print("Saved model:", path)
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
