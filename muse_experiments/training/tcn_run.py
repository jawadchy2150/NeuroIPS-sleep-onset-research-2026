"""Train the causal temporal CNN or the 5-second CNN baseline."""

import hashlib
import json
import random
import sys
import time
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
from muse_experiments.evaluation.calibration import apply_offset
from muse_experiments.evaluation.metrics import (
    bin_ids,
    range_errors,
    weighted_bmae,
)
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.temporal import (
    SleepOnsetNet,
    build_sleep_onset_net,
)
from muse_experiments.training.recording_loss import (
    chunk_loss,
    chunk_recordings,
    recording_groups,
)
from muse_experiments.preprocessing.windows import (
    SequenceDataset,
    ensure_window_cache,
    load_feature_table,
)
from muse_experiments.splits import assert_disjoint
from muse_experiments.training.lightgbm_run import split_subjects


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_scale(targets, edges, bin_weights):
    ids = bin_ids(np.asarray(targets), edges)
    values = np.asarray(bin_weights, dtype=np.float64)[ids]
    return float(values.mean())


def _batch_weights(targets, edges, bin_weights, scale):
    ids = bin_ids(targets.detach().cpu().numpy(), edges)
    values = np.asarray(bin_weights, dtype=np.float32)[ids] / scale
    return torch.as_tensor(values, device=targets.device)


def _run_epoch(model, loader, device, optimizer, cfg, scale):
    training = optimizer is not None
    model.train(training)
    edges = cfg["evaluation"]["bins"]
    bin_weights = cfg["evaluation"]["bin_weights"]
    total = 0.0
    seen = 0
    predictions = []
    targets = []
    subjects = []
    sessions = []
    for sequence, mask, target, subject, session in loader:
        sequence = sequence.to(device)
        mask = mask.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            estimate = model(sequence, mask)
        else:
            with torch.no_grad():
                estimate = model.predict(sequence, mask)
        weight = _batch_weights(target, edges, bin_weights, scale)
        loss = (torch.abs(estimate - target) * weight).mean()
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["training"]["grad_clip"]),
            )
            optimizer.step()
        batch = int(target.shape[0])
        total += float(loss.detach().cpu()) * batch
        seen += batch
        predictions.append(estimate.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        subjects.extend(subject)
        sessions.extend(session)
    return {
        "loss": total / max(seen, 1),
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "subject": subjects,
        "session": sessions,
    }


def _scores(result, cfg):
    frame = pd.DataFrame({
        "subject": result["subject"],
        "session": result["session"],
        "target_s": result["target"],
    })
    prediction = np.clip(
        result["prediction"],
        0,
        cfg["evaluation"]["cap_seconds"],
    )
    mae = float(mean_absolute_error(result["target"], prediction))
    score = weighted_bmae(
        frame,
        prediction,
        cfg["evaluation"]["bins"],
        cfg["evaluation"]["bin_weights"],
    )
    return mae, score, prediction, frame


def _loader(dataset, cfg, shuffle):
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"]["num_workers"]),
    )


def _run_recording_epoch(
    model, dataset, groups, device, optimizer, cfg, rng
):
    """One pass that updates each recording equally."""
    model.train(True)
    size = int(cfg["training"]["recordings_per_update"])
    edges = cfg["evaluation"]["bins"]
    weights = cfg["evaluation"]["bin_weights"]
    order = rng.permutation(len(groups))
    total = 0.0
    seen = 0
    for chunk in chunk_recordings(groups, order, size):
        optimizer.zero_grad(set_to_none=True)
        loss = chunk_loss(
            model, dataset, chunk, device, edges, weights
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(cfg["training"]["grad_clip"]),
        )
        optimizer.step()
        count = len(chunk)
        total += float(loss.detach().cpu()) * count
        seen += count
    if seen != len(groups):
        raise RuntimeError(
            f"updated {seen} recordings, expected {len(groups)}"
        )
    return total / seen


def _loss_name(cfg):
    return cfg["training"].get("loss", "window_weighted")


def _checkpoint(model, cfg, choice, epoch, mae, score):
    model_cfg = cfg["model"]
    return {
        "state_dict": model.state_dict(),
        "config": cfg.get("name"),
        "history_windows": int(model_cfg["history_windows"]),
        "use_temporal": bool(model_cfg["use_temporal"]),
        "width": int(model_cfg["width"]),
        "kernel_size": int(model_cfg["kernel_size"]),
        "dilations": [int(item) for item in model_cfg["dilations"]],
        "dropout": float(model_cfg["dropout"]),
        "uv_scale": float(model_cfg["uv_scale"]),
        "channels": list(cfg["data"]["channels"]),
        "cap_seconds": float(cfg["evaluation"]["cap_seconds"]),
        "split_seed": int(cfg["split"]["seed"]),
        "seed": int(cfg["seed"]),
        "epoch": int(epoch),
        "val_mae": float(mae),
        "val_wbmae": float(score),
        "device": choice.name,
        "loss": _loss_name(cfg),
    }


def fit_temporal(cfg):
    """Fit one temporal config and write a new checkpoint."""
    set_seed(int(cfg["seed"]))
    frame = load_feature_table(cfg)
    train_idx, val_idx = split_subjects(frame, cfg)
    column = cfg["split"]["group_column"]
    assert_disjoint(
        frame.iloc[train_idx][column],
        frame.iloc[val_idx][column],
    )
    print(
        "Training participants:",
        frame.iloc[train_idx][column].nunique(),
    )
    print(
        "Validation participants:",
        frame.iloc[val_idx][column].nunique(),
    )
    print("Training windows:", len(train_idx))
    print("Validation windows:", len(val_idx))
    print(
        "History windows:",
        cfg["model"]["history_windows"],
        "temporal:",
        cfg["model"]["use_temporal"],
    )

    root = repo_path(cfg["paths"]["muse_data_root"])
    cache = ensure_window_cache(frame, cfg, root)
    history = int(cfg["model"]["history_windows"])
    train_set = SequenceDataset(cache, frame, train_idx, history)
    val_set = SequenceDataset(cache, frame, val_idx, history)
    train_loader = _loader(train_set, cfg, shuffle=True)
    val_loader = _loader(val_set, cfg, shuffle=False)

    choice = resolve_device(cfg.get("device"))
    device = choice.device
    print("Torch device:", describe_device(choice))
    if cfg.get("device") == "mps" and choice.name != "mps":
        raise RuntimeError("Apple MPS was requested but is not active")
    print("Loss:", _loss_name(cfg))
    model = build_sleep_onset_net(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    edges = cfg["evaluation"]["bins"]
    bin_weights = cfg["evaluation"]["bin_weights"]
    scale = _loss_scale(
        frame.iloc[train_idx]["target_s"], edges, bin_weights
    )
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    best_mae = None
    waited = 0
    started = time.perf_counter()
    epochs = int(cfg["training"]["max_epochs"])
    patience = int(cfg["training"]["patience"])
    recording_mode = _loss_name(cfg) == "recording_wbmae"
    groups = recording_groups(train_set) if recording_mode else None
    rng = np.random.default_rng(int(cfg["seed"]))
    if recording_mode:
        size = int(cfg["training"]["recordings_per_update"])
        print(
            "Training recordings:",
            len(groups),
            "per update:",
            size,
        )
    preserved = _fingerprint() if recording_mode else None
    for epoch in range(1, epochs + 1):
        if recording_mode:
            train_loss = _run_recording_epoch(
                model, train_set, groups, device, optimizer, cfg, rng
            )
            print(f"Epoch {epoch}: train loss {train_loss:.2f} s")
        else:
            _run_epoch(model, train_loader, device, optimizer, cfg, scale)
        validation = _run_epoch(
            model, val_loader, device, None, cfg, scale
        )
        mae, score, prediction, val_frame = _scores(validation, cfg)
        print(
            f"Epoch {epoch}: MAE {mae:.2f} s, "
            f"W-bMAE {score:.2f} s"
        )
        if score < best_score:
            best_score = score
            best_mae = mae
            best_epoch = epoch
            waited = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_prediction = prediction
            best_frame = val_frame
        else:
            waited += 1
            if waited >= patience:
                print(f"Early stop at epoch {epoch}.")
                break
    elapsed = time.perf_counter() - started
    print(f"Training duration: {elapsed:.1f} seconds")
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    print(
        f"Best epoch {best_epoch}: MAE {best_mae:.2f} s, "
        f"W-bMAE {best_score:.2f} s"
    )
    for row in range_errors(
        best_frame["target_s"], best_prediction, edges
    ):
        print(
            f"  {row['range']}: MAE {row['mae']:.2f} s "
            f"({row['windows']} windows)"
        )
    _print_baselines(frame, train_idx, val_idx, edges, bin_weights)
    payload = _checkpoint(
        model, cfg, choice, best_epoch, best_mae, best_score
    )
    destination = repo_path(cfg["outputs"]["model_file"])

    def write_temp(temporary):
        torch.save(payload, temporary)

    path = publish_file(destination, write_temp)
    print("Saved model:", path)
    sidecar = path.with_suffix(".json")

    def write_json(temporary):
        summary = {
            key: value
            for key, value in payload.items()
            if key != "state_dict"
        }
        temporary.write_text(json.dumps(summary), encoding="utf-8")

    publish_file(sidecar, write_json)
    if recording_mode:
        _compare_recording_model(
            model,
            val_loader,
            frame,
            val_idx,
            device,
            cfg,
            elapsed,
            best_score,
        )
        if _fingerprint() != preserved:
            raise RuntimeError("a preserved checkpoint hash changed")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint():
    relative_paths = (
        "muse_experiments/results/models/"
        "cnn_5s_100_pre_n2_split33_seed42.pt",
        "muse_experiments/results/models/"
        "tcn_120s_100_pre_n2_split33_seed42.pt",
        "muse_experiments/results/models/"
        "lightgbm_weighted_100_pre_n2_split33.joblib",
        "muse_experiments/results/analysis/lightgbm_offset_summary.csv",
        "muse_experiments/results/analysis/lightgbm_offset_by_range.csv",
        "muse_experiments/results/analysis/lightgbm_offset_by_recording.csv",
        "muse_experiments/results/analysis/lightgbm_offset_cv.csv",
    )
    found = {}
    for relative in relative_paths:
        path = repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"preserved file missing: {path}")
        found[str(path)] = _sha256(path)
    return found


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
    return model


def _slice_bias(targets, prediction, low, high, closed):
    targets = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if closed:
        mask = (targets >= low) & (targets <= high)
    else:
        mask = (targets >= low) & (targets < high)
    error = prediction[mask] - targets[mask]
    return {
        "mae": float(np.mean(np.abs(error))),
        "bias_s": float(np.mean(error)),
        "mean_predicted_s": float(np.mean(prediction[mask])),
        "mean_actual_s": float(np.mean(targets[mask])),
        "windows": int(mask.sum()),
    }


def _metric_row(name, frame, prediction, duration=None):
    targets = frame["target_s"].to_numpy()
    mae = float(mean_absolute_error(targets, prediction))
    score = weighted_bmae(
        frame,
        prediction,
        [0, 40, 90, 300, 600],
        [10, 5, 3, 1],
    )
    near = _slice_bias(targets, prediction, 0, 40, False)
    middle = _slice_bias(targets, prediction, 40, 90, False)
    approach = _slice_bias(targets, prediction, 90, 300, False)
    distant = _slice_bias(targets, prediction, 300, 600, True)
    return {
        "model": name,
        "mae": mae,
        "wbmae": score,
        "mae_0_40": near["mae"],
        "bias_0_40": near["bias_s"],
        "mean_pred_0_40": near["mean_predicted_s"],
        "mae_40_90": middle["mae"],
        "bias_40_90": middle["bias_s"],
        "mean_pred_40_90": middle["mean_predicted_s"],
        "mae_90_300": approach["mae"],
        "bias_90_300": approach["bias_s"],
        "mae_300_600": distant["mae"],
        "bias_300_600": distant["bias_s"],
        "bias_s": float(np.mean(prediction - targets)),
        "duration_s": duration,
    }


def _print_ranges(name, targets, prediction):
    print(name)
    bands = (
        (0, 40, False, "[0, 40)"),
        (40, 90, False, "[40, 90)"),
        (90, 300, False, "[90, 300)"),
        (300, 600, True, "[300, 600]"),
    )
    for low, high, closed, label in bands:
        row = _slice_bias(targets, prediction, low, high, closed)
        print(
            f"  {label}: MAE {row['mae']:.2f} s, "
            f"mean predicted {row['mean_predicted_s']:.2f} s, "
            f"bias {row['bias_s']:.2f} s "
            f"({row['windows']} windows)"
        )


def _expect(name, score, expected):
    if abs(score - expected) > 0.02:
        raise RuntimeError(
            f"{name} W-bMAE {score:.4f} != recorded {expected:.2f}"
        )


def _compare_recording_model(
    model, loader, frame, val_idx, device, cfg, duration, best_score
):
    """Score the new CNN against the frozen baselines."""
    validation = frame.iloc[val_idx].reset_index(drop=True)
    edges = cfg["evaluation"]["bins"]
    weights = cfg["evaluation"]["bin_weights"]
    fresh = _run_epoch(model, loader, device, None, cfg, 1.0)
    _mae, score, prediction, val_frame = _scores(fresh, cfg)
    if not np.allclose(
        val_frame["target_s"], validation["target_s"], atol=1e-5
    ):
        raise RuntimeError("validation order does not match the split")
    rows = [
        _metric_row(
            "cnn_recording_loss", validation, prediction, duration
        )
    ]
    print(
        f"Corrected CNN: MAE {rows[0]['mae']:.2f} s, "
        f"W-bMAE {score:.2f} s"
    )
    if abs(score - best_score) > 0.05:
        raise RuntimeError(
            f"reloaded W-bMAE {score:.4f} != best epoch {best_score:.4f}"
        )

    original_path = repo_path(
        load_config("cnn_5s")["outputs"]["model_file"]
    )
    original = _load_net(original_path).to(device)
    original_eval = _run_epoch(
        original, loader, device, None, cfg, 1.0
    )
    _original_mae, original_score, original_pred, _frame = _scores(
        original_eval, cfg
    )
    rows.append(_metric_row("cnn_5s", validation, original_pred))
    _expect("original CNN", original_score, 76.56)

    previous_path = repo_path(
        load_config("cnn_5s_recording_loss")["outputs"]["model_file"]
    )
    if previous_path.resolve() != repo_path(
        cfg["outputs"]["model_file"]
    ).resolve():
        previous = _load_net(previous_path).to(device)
        previous_eval = _run_epoch(
            previous, loader, device, None, cfg, 1.0
        )
        _prev_mae, previous_score, previous_pred, _prev_frame = _scores(
            previous_eval, cfg
        )
        rows.append(_metric_row(
            "cnn_recording_loss_100", validation, previous_pred
        ))
        _expect("previous corrected CNN", previous_score, 57.04)
        _print_ranges(
            "Previous corrected CNN (local 100-recording run)",
            validation["target_s"].to_numpy(),
            previous_pred,
        )

    saved = joblib.load(repo_path(
        load_config("lightgbm_weighted")["inputs"]["model_file"]
    ))
    raw = np.clip(
        saved["model"].predict(validation[saved["features"]]),
        0,
        cfg["evaluation"]["cap_seconds"],
    )
    calibrated = apply_offset(raw, -16.0, cfg["evaluation"]["cap_seconds"])
    raw_score = weighted_bmae(validation, raw, edges, weights)
    calibrated_score = weighted_bmae(
        validation, calibrated, edges, weights
    )
    _expect("uncalibrated LightGBM", raw_score, 66.56)
    _expect("calibrated LightGBM", calibrated_score, 59.88)
    rows.append(_metric_row(
        "lightgbm_calibrated_-16s", validation, calibrated
    ))
    constant = np.full(len(validation), 35.0)
    constant_score = weighted_bmae(validation, constant, edges, weights)
    _expect("constant 35s", constant_score, 63.31)
    rows.append(_metric_row("constant_35s", validation, constant))

    table = pd.DataFrame(rows)
    desired = repo_path(cfg["outputs"]["comparison_csv"])

    def write_temp(path):
        table.to_csv(path, index=False)

    written = publish_file(desired, write_temp)
    print(table.to_string(index=False))
    print("Wrote", written)
    _print_ranges(
        "This corrected CNN",
        validation["target_s"].to_numpy(),
        prediction,
    )
    print(
        "Previous official Codabench W-bMAE for the "
        "100-recording corrected CNN: 60.39 s. "
        "This run was not submitted."
    )
    return table


def _print_baselines(frame, train_idx, val_idx, edges, weights):
    validation = frame.iloc[val_idx]
    median = float(np.median(frame.iloc[train_idx]["target_s"]))
    for label, constant in (
        (f"training median ({median:.1f}s)", median),
        ("constant 35s", 35.0),
    ):
        prediction = np.full(len(validation), constant)
        score = weighted_bmae(validation, prediction, edges, weights)
        mae = float(mean_absolute_error(
            validation["target_s"], prediction
        ))
        print(
            f"{label}: MAE {mae:.2f} s, W-bMAE {score:.2f} s"
        )
    print("Recorded weighted LightGBM local W-bMAE: 66.56 s")
    print("Recorded Codabench weighted LightGBM W-bMAE: 68.71 s")


def main(default_config):
    args = sys.argv[1:]
    name = args[0] if args else default_config
    fit_temporal(load_config(name))
