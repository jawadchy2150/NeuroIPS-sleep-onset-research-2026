"""Calibrate the saved weighted LightGBM with a constant offset.

The offset is selected on training recordings only. The 17 validation
participants are scored after that choice is fixed. Inner cross-validation
also stays inside the training participants. Nothing is written over the
saved checkpoint, booster, or Codabench submission.
"""

import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from muse_experiments.config import load_config
from muse_experiments.evaluation.calibration import (
    apply_offset,
    offset_grid,
    select_offset,
)
from muse_experiments.evaluation.metrics import (
    bin_ids,
    bin_sample_weights,
    weighted_bmae,
)
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.lightgbm import build_regressor
from muse_experiments.preprocessing.windows import load_feature_table
from muse_experiments.splits import assert_disjoint
from muse_experiments.training.lightgbm_run import (
    feature_frame,
    split_subjects,
)

ANALYSIS = "muse_experiments/results/analysis"
EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
WEIGHTS = (10.0, 5.0, 3.0, 1.0)
CAP = 600.0
CV_SPLITS = 5
CV_SEED = 33
EXPECTED_OFFSET = -16.0
EXPECTED_VAL_WBMAE = 59.88
PRESERVED = (
    "muse_experiments/results/models/"
    "lightgbm_weighted_100_pre_n2_split33.joblib",
    "muse_experiments/results/models/"
    "lightgbm_weighted_100_pre_n2_split33_booster.json",
    "muse_experiments/results/submissions/"
    "muse_weighted_lightgbm_pre_n2.zip",
    "muse_experiments/results/models/"
    "cnn_5s_100_pre_n2_split33_seed42.pt",
    "muse_experiments/results/models/"
    "tcn_120s_100_pre_n2_split33_seed42.pt",
)


def _edges():
    return list(EDGES)


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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(paths):
    found = {}
    for relative in paths:
        path = repo_path(relative)
        if path.is_file():
            found[str(path)] = _sha256(path)
    return found


def _predict(model, frame, features):
    raw = model.predict(frame[features])
    return np.clip(np.asarray(raw, dtype=np.float64), 0.0, CAP)


def _write_csv(frame, name):
    desired = repo_path(f"{ANALYSIS}/{name}")

    def write_temp(path):
        frame.to_csv(path, index=False)

    return publish_file(desired, write_temp)


def _range_rows(name, targets, prediction):
    rows = []
    for index, mask in enumerate(_masks(targets)):
        if not np.any(mask):
            continue
        actual = targets[mask]
        estimate = prediction[mask]
        rows.append({
            "prediction": name,
            "range": _range_label(index),
            "windows": int(mask.sum()),
            "mae": float(np.mean(np.abs(estimate - actual))),
            "mean_predicted_s": float(np.mean(estimate)),
            "mean_actual_s": float(np.mean(actual)),
            "bias_s": float(np.mean(estimate - actual)),
        })
    return rows


def _recording_rows(split_name, frame, raw, calibrated):
    rows = []
    grouped = frame.groupby(["subject", "session"], sort=False)
    for (subject, session), part in grouped:
        positions = part.index.to_numpy()
        # ``frame`` is reset, so positions are positional.
        actual = part["target_s"].to_numpy()
        raw_part = raw[positions]
        cal_part = calibrated[positions]
        rows.append({
            "split": split_name,
            "subject": subject,
            "session": session,
            "windows": int(len(part)),
            "wbmae_raw": weighted_bmae(part, raw_part, _edges(), WEIGHTS),
            "wbmae_calibrated": weighted_bmae(
                part, cal_part, _edges(), WEIGHTS
            ),
            "mae_raw": float(mean_absolute_error(actual, raw_part)),
            "mae_calibrated": float(
                mean_absolute_error(actual, cal_part)
            ),
            "bias_raw_s": float(np.mean(raw_part - actual)),
            "bias_calibrated_s": float(np.mean(cal_part - actual)),
        })
    return rows


def _subject_folds(subjects):
    unique = np.array(sorted(set(np.asarray(subjects).tolist())))
    splitter = KFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=CV_SEED,
    )
    folds = []
    for train_pos, held_pos in splitter.split(unique):
        folds.append((
            set(unique[train_pos].tolist()),
            set(unique[held_pos].tolist()),
        ))
    return folds


def _mask_subjects(frame, subjects):
    return frame["subject"].isin(subjects).to_numpy()


def _fit_fold(features, train_frame, cfg):
    model = build_regressor(cfg)
    weights = bin_sample_weights(
        train_frame, _edges(), list(WEIGHTS)
    )
    model.fit(
        train_frame[features],
        train_frame["target_s"],
        sample_weight=weights,
    )
    return model


def _cv_rows(frame, features, frozen_prediction, cfg):
    rows = []
    grid = offset_grid()
    for fold, (train_ids, held_ids) in enumerate(
        _subject_folds(frame["subject"]), start=1
    ):
        assert_disjoint(list(train_ids), list(held_ids))
        train_mask = _mask_subjects(frame, train_ids)
        held_mask = _mask_subjects(frame, held_ids)
        train_part = frame.loc[train_mask]
        held_part = frame.loc[held_mask]
        protocols = [("frozen_predictions", frozen_prediction)]
        refit = _fit_fold(features, train_part, cfg)
        protocols.append((
            "refit_model",
            _predict(refit, frame, features),
        ))
        for protocol, prediction in protocols:
            offset, train_score, _curve = select_offset(
                train_part,
                prediction[train_mask],
                _edges(),
                WEIGHTS,
                grid,
                CAP,
            )
            held_raw = prediction[held_mask]
            held_cal = apply_offset(held_raw, offset, CAP)
            rows.append({
                "protocol": protocol,
                "fold": fold,
                "fit_subjects": len(train_ids),
                "held_subjects": len(held_ids),
                "offset_s": offset,
                "fit_wbmae": train_score,
                "held_wbmae_raw": weighted_bmae(
                    held_part, held_raw, _edges(), WEIGHTS
                ),
                "held_wbmae_calibrated": weighted_bmae(
                    held_part, held_cal, _edges(), WEIGHTS
                ),
                "held_mae_calibrated": float(mean_absolute_error(
                    held_part["target_s"], held_cal
                )),
            })
    return rows


def _present_bins(frame):
    ident = bin_ids(frame["target_s"].to_numpy(), _edges())
    labeled = frame.assign(_bin=ident)
    counts = labeled.groupby(
        ["subject", "session"]
    )["_bin"].nunique()
    return int((counts < 4).sum()), int(len(counts))


def calibrate():
    before = _fingerprint(PRESERVED)
    cfg = load_config("lightgbm_weighted")
    frame = load_feature_table(cfg)
    _features, feature_names = feature_frame(frame)
    train_idx, val_idx = split_subjects(frame, cfg)
    checkpoint = repo_path(cfg["inputs"]["model_file"])
    saved = joblib.load(checkpoint)
    if list(saved["features"]) != feature_names:
        raise RuntimeError("checkpoint feature order does not match")
    model = saved["model"]
    raw = _predict(model, frame, saved["features"])

    training = frame.iloc[train_idx].reset_index(drop=True)
    validation = frame.iloc[val_idx].reset_index(drop=True)
    if validation["subject"].nunique() != 17 or len(validation) != 2603:
        raise RuntimeError("validation split is not the recorded 17 / 2603")
    if int(validation["has_official_test"].astype(int).sum()) != 0:
        raise RuntimeError("validation includes an official test subject")
    assert_disjoint(training["subject"], validation["subject"])
    train_raw = raw[train_idx]
    val_raw = raw[val_idx]
    raw_val_mae = float(mean_absolute_error(
        validation["target_s"], val_raw
    ))
    raw_val_score = weighted_bmae(
        validation, val_raw, _edges(), WEIGHTS
    )
    if abs(raw_val_score - 66.56041645928337) > 1e-6:
        raise RuntimeError(
            f"uncalibrated W-bMAE changed: {raw_val_score}"
        )

    grid = offset_grid()
    offset, train_score, curve = select_offset(
        training, train_raw, _edges(), WEIGHTS, grid, CAP
    )
    if abs(offset - EXPECTED_OFFSET) > 1e-9:
        raise RuntimeError(
            f"selected offset {offset} != {EXPECTED_OFFSET}"
        )
    val_cal = apply_offset(val_raw, offset, CAP)
    train_cal = apply_offset(train_raw, offset, CAP)
    val_score = weighted_bmae(validation, val_cal, _edges(), WEIGHTS)
    val_mae = float(mean_absolute_error(
        validation["target_s"], val_cal
    ))
    if abs(val_score - EXPECTED_VAL_WBMAE) > 0.02:
        raise RuntimeError(
            f"calibrated validation W-bMAE {val_score} != "
            f"{EXPECTED_VAL_WBMAE}"
        )
    curve_scores = [score for _item, score in curve]
    minimums = [
        item for item, score in curve
        if abs(score - min(curve_scores)) <= 1e-12
    ]

    print(f"Offset selected on training only: {offset:.1f} s")
    print(f"Grid minima: {minimums}")
    train_raw_score = weighted_bmae(
        training, train_raw, _edges(), WEIGHTS
    )
    print(
        f"Training W-bMAE {train_raw_score:.2f} -> {train_score:.2f}"
    )
    print(
        f"Validation W-bMAE {raw_val_score:.2f} -> {val_score:.2f}"
    )
    print(f"Validation MAE {raw_val_mae:.2f} -> {val_mae:.2f}")
    missing, recordings = _present_bins(training)
    print(
        f"Training recordings missing a target bin: "
        f"{missing} of {recordings}"
    )

    summary = pd.DataFrame([
        {
            "offset_s": offset,
            "grid_minima": " ".join(f"{item:.1f}" for item in minimums),
            "train_wbmae_raw": train_raw_score,
            "train_wbmae_calibrated": train_score,
            "train_mae_raw": float(mean_absolute_error(
                training["target_s"], train_raw
            )),
            "train_mae_calibrated": float(mean_absolute_error(
                training["target_s"], train_cal
            )),
            "val_wbmae_raw": raw_val_score,
            "val_wbmae_calibrated": val_score,
            "val_mae_raw": raw_val_mae,
            "val_mae_calibrated": val_mae,
        }
    ])
    ranges = pd.DataFrame(
        _range_rows("uncalibrated", validation["target_s"], val_raw)
        + _range_rows("calibrated", validation["target_s"], val_cal)
    )
    recordings_table = pd.DataFrame(
        _recording_rows("validation", validation, val_raw, val_cal)
        + _recording_rows("training", training, train_raw, train_cal)
    )
    print("Inner cross-validation on training participants...")
    cv = pd.DataFrame(_cv_rows(
        training, list(saved["features"]), train_raw, cfg
    ))
    for protocol, part in cv.groupby("protocol", sort=False):
        print(
            protocol,
            "offsets",
            part["offset_s"].tolist(),
            "mean held W-bMAE",
            round(part["held_wbmae_calibrated"].mean(), 2),
        )
    curve_frame = pd.DataFrame(curve, columns=["offset_s", "train_wbmae"])
    written = [
        _write_csv(summary, "lightgbm_offset_summary.csv"),
        _write_csv(ranges, "lightgbm_offset_by_range.csv"),
        _write_csv(recordings_table, "lightgbm_offset_by_recording.csv"),
        _write_csv(cv, "lightgbm_offset_cv.csv"),
        _write_csv(curve_frame, "lightgbm_offset_curve.csv"),
    ]
    after = _fingerprint(PRESERVED)
    if before != after:
        raise RuntimeError("a preserved checkpoint hash changed")
    for path in written:
        print("Wrote", path)
    return summary, ranges, recordings_table, cv


def main():
    calibrate()


if __name__ == "__main__":
    main()
