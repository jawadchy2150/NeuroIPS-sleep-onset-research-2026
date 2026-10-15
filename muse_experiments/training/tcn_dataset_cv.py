"""Train the causal 120-second TCN on the saved 250-recording folds.

The participant assignments and the 250-recording selections come from
the completed CNN cross-validation. The locked 17 participants stay out.
"""

import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import weighted_bmae
from muse_experiments.io import REPO_ROOT, repo_path
from muse_experiments.models.temporal import build_sleep_onset_net
from muse_experiments.preprocessing.build_cv_pool import prepare_cv_pool
from muse_experiments.preprocessing.windows import SequenceDataset
from muse_experiments.splits import assert_disjoint, select_training_recordings
from muse_experiments.training.dataset_size_cv import (
    CV_SEED,
    _indices,
    _metrics,
    _open_cache,
    _pairs,
    _rss_bytes,
    _write_json,
)
from muse_experiments.training.recording_loss import recording_groups
from muse_experiments.training.tcn_run import (
    _loader,
    _run_epoch,
    _run_recording_epoch,
    _scores,
    set_seed,
)

CNN_DIR = REPO_ROOT / "muse_experiments" / "results" / "cv_dataset_size"
OUTPUT_DIR = REPO_ROOT / "muse_experiments" / "results" / "cv_tcn_120s"
MODEL_DIR = (
    REPO_ROOT / "muse_experiments" / "results" / "models" / "cv_tcn_120s"
)
EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]
MEMORY_LIMIT_BYTES = 8 * 1024 ** 3
PROTECTED = (
    "muse_experiments/results/cv_dataset_size/assignments.csv",
    "muse_experiments/results/cv_dataset_size/fold_scores.csv",
    "muse_experiments/results/cv_dataset_size/subject_scores.csv",
    "muse_experiments/results/features/"
    "muse_features_cv_pool_locked17_pre_n2.csv",
    "muse_experiments/results/tcn/windows_cv_pool_locked17_pre_n2.dat",
    "muse_experiments/results/models/"
    "cnn_5s_recording_loss_250train_locked17_seed42.pt",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_assignment():
    saved = pd.read_csv(CNN_DIR / "assignments.csv")
    return {
        str(subject): int(fold)
        for subject, fold in zip(saved["subject"], saved["fold"])
    }


def _cnn_reference(fold):
    path = CNN_DIR / f"fold_{fold}_n250.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _tcn_config():
    cfg = load_config("tcn_120s")
    cfg["device"] = "mps"
    cfg["training"]["loss"] = "recording_wbmae"
    cfg["training"]["recordings_per_update"] = 1
    if int(cfg["model"]["history_windows"]) != 24:
        raise RuntimeError("TCN history is not 24 windows")
    if not cfg["model"]["use_temporal"]:
        raise RuntimeError("TCN temporal stack is disabled")
    if int(cfg["training"]["max_epochs"]) != 15:
        raise RuntimeError("epoch budget is not 15")
    if float(cfg["training"]["learning_rate"]) != 0.001:
        raise RuntimeError("learning rate is not 0.001")
    return cfg


def _selection(frame, pairs, assignment, fold):
    val_subjects = [
        subject for subject, label in assignment.items() if label == fold
    ]
    chosen = select_training_recordings(pairs, val_subjects, 250)
    train_idx = _indices(frame, chosen)
    val_pairs = [pair for pair in pairs if pair[0] in set(val_subjects)]
    val_idx = _indices(frame, val_pairs)
    assert_disjoint(
        frame.iloc[train_idx]["subject"],
        frame.iloc[val_idx]["subject"],
    )
    reference = _cnn_reference(fold)
    train_subjects = frame.iloc[train_idx]["subject"].nunique()
    val_groups = frame.iloc[val_idx].groupby(["subject", "session"])
    val_recordings = val_groups.ngroups
    if len(train_idx) != int(reference["train_windows"]):
        raise RuntimeError(
            f"fold {fold} train windows {len(train_idx)} "
            f"!= CNN {reference['train_windows']}"
        )
    if train_subjects != int(reference["train_subjects"]):
        raise RuntimeError(f"fold {fold} training participants differ")
    if val_recordings != int(reference["val_recordings"]):
        raise RuntimeError(f"fold {fold} validation recordings differ")
    if int(reference["val_subjects"]) != len(val_subjects):
        raise RuntimeError(f"fold {fold} validation participants differ")
    return train_idx, val_idx


def _fit(frame, cache, train_idx, val_idx, cfg, fold):
    set_seed(CV_SEED)
    choice = resolve_device(cfg.get("device"))
    if choice.name != "mps":
        raise RuntimeError("Apple MPS was requested but is not active")
    device = choice.device
    print("Torch device:", describe_device(choice))
    history = int(cfg["model"]["history_windows"])
    train_set = SequenceDataset(cache, frame, train_idx, history)
    val_set = SequenceDataset(cache, frame, val_idx, history)
    groups = recording_groups(train_set)
    if len(groups) != 250:
        raise RuntimeError(f"grouped {len(groups)} recordings, expected 250")
    if int(cfg["training"]["recordings_per_update"]) != 1:
        raise RuntimeError("recordings must be updated one at a time")
    val_loader = _loader(val_set, cfg, shuffle=False)
    model = build_sleep_onset_net(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    rng = np.random.default_rng(CV_SEED)
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    best_result = None
    best_train = None
    waited = 0
    curves = []
    started = time.perf_counter()
    epochs = int(cfg["training"]["max_epochs"])
    patience = int(cfg["training"]["patience"])
    for epoch in range(1, epochs + 1):
        train_loss = _run_recording_epoch(
            model, train_set, groups, device, optimizer, cfg, rng
        )
        validation = _run_epoch(model, val_loader, device, None, cfg, 1.0)
        mae, score, prediction, val_frame = _scores(validation, cfg)
        curves.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_mae": float(mae),
            "val_wbmae": float(score),
        })
        print(
            f"Fold {fold} TCN epoch {epoch}: "
            f"train {train_loss:.2f} s, W-bMAE {score:.2f} s"
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            waited = 0
            best_train = float(train_loss)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_result = (mae, score, prediction, val_frame)
        else:
            waited += 1
            if waited >= patience:
                print(f"Early stop at epoch {epoch}.")
                break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    _mae, score, prediction, val_frame = best_result
    destination = MODEL_DIR / f"tcn_fold_{fold}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        torch.save({
            "state_dict": model.state_dict(),
            "config": "tcn_120s_recording_loss_cv",
            "fold": int(fold),
            "training_recordings": 250,
            "history_windows": history,
            "use_temporal": True,
            "cv_seed": CV_SEED,
            "epoch": int(best_epoch),
            "val_wbmae": float(score),
            "loss": "recording_wbmae",
            "device": choice.name,
        }, destination)
    metrics = _metrics(
        fold, 250, frame, train_idx, val_frame, prediction,
        best_epoch, elapsed, _rss_bytes(),
    )
    metrics["train_wbmae"] = float(best_train)
    metrics["curves"] = curves
    metrics["checkpoint"] = str(destination.relative_to(REPO_ROOT))
    del model, optimizer, best_state
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return metrics, prediction, val_frame


def _subject_rows(fold, prediction, val_frame, cnn_subjects):
    rows = []
    cnn = cnn_subjects[cnn_subjects["fold"] == fold]
    lookup = {
        str(row.subject): float(row.wbmae_250)
        for row in cnn.itertuples(index=False)
    }
    for subject in sorted(set(val_frame["subject"].astype(str))):
        mask = val_frame["subject"].astype(str).to_numpy() == subject
        score = weighted_bmae(
            val_frame.loc[mask], prediction[mask], EDGES, WEIGHTS
        )
        cnn_score = lookup[subject]
        rows.append({
            "fold": fold,
            "subject": subject,
            "wbmae_cnn_250": cnn_score,
            "wbmae_tcn": score,
            "improvement_s": cnn_score - score,
        })
    return rows


def _save(results, subject_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = []
    curves = []
    for metrics in results:
        row = {
            key: value for key, value in metrics.items() if key != "curves"
        }
        flat.append(row)
        for point in metrics["curves"]:
            curves.append({
                "fold": metrics["fold"],
                **point,
            })
    pd.DataFrame(flat).to_csv(OUTPUT_DIR / "fold_scores.csv", index=False)
    pd.DataFrame(curves).to_csv(OUTPUT_DIR / "curves.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(
        OUTPUT_DIR / "subject_scores.csv", index=False
    )
    cnn = pd.read_csv(CNN_DIR / "summary.csv")
    rows = []
    for fold in range(1, 6):
        tcn = next(item for item in results if item["fold"] == fold)
        cnn_score = float(cnn.loc[cnn["fold"] == fold, "wbmae_250"].iloc[0])
        rows.append({
            "fold": fold,
            "wbmae_cnn_250": cnn_score,
            "wbmae_tcn": tcn["wbmae"],
            "improvement_s": cnn_score - tcn["wbmae"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    return summary


def main():
    before = {item: _sha256(REPO_ROOT / item) for item in PROTECTED}
    cfg = _tcn_config()
    assignment = _load_assignment()
    frame, cache_path, locked = prepare_cv_pool()
    if set(frame["subject"].astype(str)) & set(locked):
        raise RuntimeError("the CV pool contains a locked subject")
    cache = _open_cache(cache_path, len(frame))
    pairs = _pairs(frame)
    for fold in range(1, 6):
        _selection(frame, pairs, assignment, fold)
    print("Fold selections match the 250-recording CNN.")
    cnn_subjects = pd.read_csv(CNN_DIR / "subject_scores.csv")
    results = []
    subject_rows = []
    for fold in range(1, 6):
        path = OUTPUT_DIR / f"fold_{fold}.json"
        train_idx, val_idx = _selection(frame, pairs, assignment, fold)
        if path.is_file():
            print(f"Fold {fold}: reusing {path.name}")
            saved = json.loads(path.read_text(encoding="utf-8"))
            results.append(saved)
            continue
        print(
            f"Fold {fold}: train windows {len(train_idx)}, "
            f"validation windows {len(val_idx)}"
        )
        metrics, prediction, val_frame = _fit(
            frame, cache, train_idx, val_idx, cfg, fold
        )
        stored = dict(metrics)
        curves = stored.pop("curves")
        _write_json(path, stored | {"curves": curves})
        results.append(stored | {"curves": curves})
        subject_rows.extend(
            _subject_rows(fold, prediction, val_frame, cnn_subjects)
        )
        rss = int(metrics["peak_rss_bytes"])
        print(f"Process peak RSS: {rss / (1024 ** 3):.2f} GiB")
        if fold == 1 and rss > MEMORY_LIMIT_BYTES:
            raise RuntimeError(
                "fold 1 exceeded 8 GiB; remaining folds were not started"
            )
    if len(subject_rows) < 186 and (
        OUTPUT_DIR / "subject_scores.csv"
    ).is_file():
        subject_rows = pd.read_csv(
            OUTPUT_DIR / "subject_scores.csv"
        ).to_dict("records")
    summary = _save(results, subject_rows)
    print(summary.to_string(index=False))
    tcn_scores = [item["wbmae"] for item in results]
    print(
        "Mean TCN W-bMAE: "
        f"{float(np.mean(tcn_scores)):.2f} ± "
        f"{float(np.std(tcn_scores, ddof=1)):.2f} s"
    )
    gains = summary["improvement_s"].to_numpy()
    print(
        "Mean improvement over CNN: "
        f"{float(np.mean(gains)):.2f} ± "
        f"{float(np.std(gains, ddof=1)):.2f} s"
    )
    print(
        "Folds won by TCN: "
        f"{int((gains > 0).sum())} of 5"
    )
    after = {item: _sha256(REPO_ROOT / item) for item in PROTECTED}
    if after != before:
        raise RuntimeError("a protected research file changed")
    print("Protected CNN results were unchanged.")


if __name__ == "__main__":
    main()
