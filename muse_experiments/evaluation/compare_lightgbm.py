"""Compare corrected LightGBM models with constant baselines.

Reads the pre-N2 feature table and the two saved checkpoints. It does not
train, and it does not read the earlier post-N2 table or joblib file.
"""

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
from muse_experiments.evaluation.metrics import range_errors, weighted_bmae
from muse_experiments.io import repo_path
from muse_experiments.training.lightgbm_run import split_subjects


def _score(name, frame, prediction, bins, weights):
    mae = mean_absolute_error(frame["target_s"], prediction)
    score = weighted_bmae(frame, prediction, bins, weights)
    print(f"\n{name}")
    print("  MAE:", round(mae, 2), "seconds")
    print("  W-bMAE:", round(score, 2), "seconds")
    for row in range_errors(frame["target_s"], prediction, bins):
        print(
            f"  {row['range']}: MAE {row['mae']:.2f} s "
            f"({row['windows']} windows)"
        )
    return {"name": name, "mae": mae, "wbmae": score}


def compare(frame, plain_cfg, weighted_cfg):
    bins = plain_cfg["evaluation"]["bins"]
    weights = plain_cfg["evaluation"]["bin_weights"]
    cap = plain_cfg["evaluation"]["cap_seconds"]
    train_idx, val_idx = split_subjects(frame, plain_cfg)
    validation = frame.iloc[val_idx].copy()
    y_train = frame.iloc[train_idx]["target_s"]
    rows = []
    for label, cfg in (
        ("unweighted LightGBM", plain_cfg),
        ("weighted LightGBM", weighted_cfg),
    ):
        path = repo_path(cfg["inputs"]["model_file"])
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        saved = joblib.load(path)
        prediction = np.clip(
            saved["model"].predict(validation[saved["features"]]),
            0,
            cap,
        )
        rows.append(_score(label, validation, prediction, bins, weights))
    median_value = float(np.median(y_train))
    for label, constant in (
        (f"training median ({median_value:.1f}s)", median_value),
        ("constant 35s", 35.0),
    ):
        prediction = np.full(len(validation), constant)
        rows.append(_score(label, validation, prediction, bins, weights))
    return pd.DataFrame(rows)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    plain = load_config(args[0] if args else "lightgbm_100")
    weighted = load_config("lightgbm_weighted")
    features_csv = repo_path(plain["inputs"]["features_csv"])
    if not features_csv.is_file():
        raise FileNotFoundError(
            f"corrected feature table not found: {features_csv}"
        )
    frame = pd.read_csv(features_csv)
    compare(frame, plain, weighted)


if __name__ == "__main__":
    main()
