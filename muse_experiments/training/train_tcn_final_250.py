"""Train the 120-second TCN for a fixed 14 epochs.

The training recordings are the 250 used by the locked-holdout CNN.
The original 17 participants are scored only after training.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import weighted_bmae
from muse_experiments.io import REPO_ROOT, repo_path
from muse_experiments.models.temporal import (
    SleepOnsetNet,
    build_sleep_onset_net,
)
from muse_experiments.preprocessing.windows import (
    SequenceDataset,
    ensure_window_cache,
    load_feature_table,
)
from muse_experiments.training.dataset_size_cv import _band, _rss_bytes
from muse_experiments.training.lightgbm_run import split_subjects
from muse_experiments.training.recording_loss import recording_groups
from muse_experiments.training.tcn_run import (
    _loader,
    _run_epoch,
    _run_recording_epoch,
    _scores,
    set_seed,
)

EPOCHS = 14
SEED = 42
EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]
MODEL_PATH = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "tcn_120s_recording_loss_250train_locked17_epoch14_seed42.pt"
)
REPORT_PATH = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "analysis"
    / "tcn_120s_250_locked17_eval.json"
)
PROTECTED = (
    "muse_experiments/results/models/"
    "cnn_5s_recording_loss_250train_locked17_seed42.pt",
    "muse_experiments/results/features/"
    "muse_features_250train_locked17_pre_n2.csv",
    "muse_experiments/results/tcn/windows_250train_locked17_pre_n2.dat",
    "muse_experiments/results/submissions/muse_cnn_250_recordings.zip",
    "muse_experiments/results/submissions/muse_cnn_recording_loss.zip",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings():
    cfg = load_config("tcn_120s")
    cfg["device"] = "mps"
    cfg["seed"] = SEED
    cfg["training"]["loss"] = "recording_wbmae"
    cfg["training"]["recordings_per_update"] = 1
    cfg["training"]["max_epochs"] = EPOCHS
    if int(cfg["model"]["history_windows"]) != 24:
        raise RuntimeError("TCN history is not 24 windows")
    if not bool(cfg["model"]["use_temporal"]):
        raise RuntimeError("TCN temporal stack is disabled")
    if float(cfg["training"]["learning_rate"]) != 0.001:
        raise RuntimeError("learning rate is not 0.001")
    return cfg


def _check_split(frame, train_idx, val_idx, data_cfg):
    locked = {
        str(subject)
        for subject in data_cfg["split"]["validation_subjects"]
    }
    train = frame.iloc[train_idx]
    val = frame.iloc[val_idx]
    train_subjects = set(train["subject"].astype(str))
    val_subjects = set(val["subject"].astype(str))
    if train_subjects & val_subjects:
        raise RuntimeError("training and validation subjects overlap")
    if train_subjects & locked:
        raise RuntimeError("a locked participant is in training")
    if val_subjects != locked:
        raise RuntimeError("validation subjects are not the locked 17")
    train_recordings = train.groupby(["subject", "session"]).ngroups
    val_recordings = val.groupby(["subject", "session"]).ngroups
    if train_recordings != 250 or val_recordings != 17:
        raise RuntimeError(
            f"expected 250/17 recordings, found "
            f"{train_recordings}/{val_recordings}"
        )
    if len(train_idx) != 37854 or len(val_idx) != 2603:
        raise RuntimeError(
            f"unexpected window counts {len(train_idx)} {len(val_idx)}"
        )
    print(
        "Training recordings: 250, subjects:",
        len(train_subjects),
        "windows:",
        len(train_idx),
    )
    print("Held-out recordings: 17, windows:", len(val_idx))


def _score(model, frame, cache, indices, cfg):
    dataset = SequenceDataset(
        cache, frame, indices, int(cfg["model"]["history_windows"])
    )
    loader = _loader(dataset, cfg, shuffle=False)
    device = next(model.parameters()).device
    result = _run_epoch(model, loader, device, None, cfg, 1.0)
    mae, score, prediction, val_frame = _scores(result, cfg)
    targets = val_frame["target_s"].to_numpy()
    near = _band(targets, prediction, 0, 40, False)
    middle = _band(targets, prediction, 40, 90, False)
    return {
        "wbmae": float(score),
        "mae": float(mae),
        "mae_0_40": near["mae"],
        "bias_0_40": near["bias_s"],
        "mean_pred_0_40": near["mean_predicted_s"],
        "mae_40_90": middle["mae"],
        "bias_40_90": middle["bias_s"],
        "mean_pred_40_90": middle["mean_predicted_s"],
    }


def _cnn_reference(frame, cache, val_idx, cfg, device):
    path = repo_path(
        load_config("cnn_5s_recording_loss_250")["outputs"]["model_file"]
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SleepOnsetNet(
        n_channels=4,
        width=32,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
        uv_scale=100.0,
        use_temporal=False,
        cap_seconds=600.0,
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    # The reference CNN scores one window. History stays at 1.
    cnn_cfg = dict(cfg)
    cnn_cfg["model"] = dict(cfg["model"])
    cnn_cfg["model"]["history_windows"] = 1
    model.eval()
    return _score(model, frame, cache, val_idx, cnn_cfg)


def _train(frame, cache, train_idx, cfg):
    set_seed(SEED)
    choice = resolve_device("mps")
    if choice.name != "mps":
        raise RuntimeError("Apple MPS was requested but is not active")
    device = choice.device
    print("Torch device:", describe_device(choice))
    dataset = SequenceDataset(
        cache, frame, train_idx, int(cfg["model"]["history_windows"])
    )
    groups = recording_groups(dataset)
    if len(groups) != 250:
        raise RuntimeError(f"grouped {len(groups)} recordings, expected 250")
    model = build_sleep_onset_net(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = _run_recording_epoch(
            model, dataset, groups, device, optimizer, cfg, rng
        )
        losses.append(float(loss))
        print(f"Epoch {epoch}: train loss {loss:.2f} s")
    elapsed = time.perf_counter() - started
    if MODEL_PATH.exists():
        raise FileExistsError(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": "tcn_120s_recording_loss_250_epoch14",
        "history_windows": 24,
        "use_temporal": True,
        "seed": SEED,
        "epoch": EPOCHS,
        "loss": "recording_wbmae",
        "device": choice.name,
        "train_loss": losses,
    }, MODEL_PATH)
    print("Saved model:", MODEL_PATH)
    print(f"Training duration: {elapsed:.1f} s")
    print(f"Peak RSS: {_rss_bytes() / (1024 ** 3):.2f} GiB")
    return model, elapsed, losses


def main():
    before = {
        item: _sha256(REPO_ROOT / item) for item in PROTECTED
    }
    data_cfg = load_config("cnn_5s_recording_loss_250")
    cfg = _settings()
    frame = load_feature_table(data_cfg)
    train_idx, val_idx = split_subjects(frame, data_cfg)
    _check_split(frame, train_idx, val_idx, data_cfg)
    root = repo_path(data_cfg["paths"]["muse_data_root"])
    cache = ensure_window_cache(frame, data_cfg, root)
    model, elapsed, losses = _train(frame, cache, train_idx, cfg)
    model.eval()
    tcn = _score(model, frame, cache, val_idx, cfg)
    cnn = _cnn_reference(
        frame, cache, val_idx, cfg, next(model.parameters()).device
    )
    report = {
        "epochs": EPOCHS,
        "duration_s": elapsed,
        "peak_rss_bytes": _rss_bytes(),
        "train_loss": losses,
        "tcn": tcn,
        "cnn_250": cnn,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise FileExistsError(REPORT_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"TCN holdout W-bMAE {tcn['wbmae']:.2f} s, "
        f"MAE {tcn['mae']:.2f} s"
    )
    print(
        f"CNN 250 holdout W-bMAE {cnn['wbmae']:.2f} s, "
        f"MAE {cnn['mae']:.2f} s"
    )
    print(
        "0-40 s TCN MAE "
        f"{tcn['mae_0_40']:.2f}, bias {tcn['bias_0_40']:.2f}"
    )
    print(
        "40-90 s TCN MAE "
        f"{tcn['mae_40_90']:.2f}, bias {tcn['bias_40_90']:.2f}"
    )
    after = {
        item: _sha256(REPO_ROOT / item) for item in PROTECTED
    }
    if after != before:
        raise RuntimeError("a protected file changed")
    print("Protected files were unchanged.")


if __name__ == "__main__":
    main()
