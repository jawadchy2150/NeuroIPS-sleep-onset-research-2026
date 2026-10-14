"""Train the causal temporal CNN or the 5-second CNN baseline."""

import json
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import (
    bin_ids,
    range_errors,
    weighted_bmae,
)
from muse_experiments.io import publish_file, repo_path
from muse_experiments.models.temporal import build_sleep_onset_net
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
    for epoch in range(1, epochs + 1):
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
    return path


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
