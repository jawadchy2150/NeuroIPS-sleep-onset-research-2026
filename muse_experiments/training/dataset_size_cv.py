"""Subject-disjoint comparison of 100 and 250 training recordings.

The original 17 validation participants stay out of every fold. Official
test sessions are not read. Each fold trains the same 5-second CNN twice,
and the 100-recording set is the first 100 recordings of that fold's
250-recording set.
"""

import gc
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from muse_experiments.config import load_config
from muse_experiments.device import describe_device, resolve_device
from muse_experiments.evaluation.metrics import weighted_bmae
from muse_experiments.io import REPO_ROOT, repo_path
from muse_experiments.models.temporal import build_sleep_onset_net
from muse_experiments.preprocessing.build_cv_pool import (
    POOL_CACHE,
    inventory_training_pool,
    locked_holdout_subjects,
    prepare_cv_pool,
)
from muse_experiments.preprocessing.windows import WindowCache
from muse_experiments.splits import (
    assert_disjoint,
    assign_subject_folds,
    select_training_recordings,
)
from muse_experiments.training.recording_loss import recording_groups
from muse_experiments.training.tcn_run import (
    _loader,
    _run_epoch,
    _run_recording_epoch,
    _scores,
    set_seed,
)

CV_SEED = 42
SIZES = (100, 250)
OUTPUT_DIR = REPO_ROOT / "muse_experiments" / "results" / "cv_dataset_size"
MODEL_DIR = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "cv_dataset_size"
)
EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]
PROTECTED = (
    "muse_experiments/results/models/"
    "cnn_5s_recording_loss_100_pre_n2_split33_seed42.pt",
    "muse_experiments/results/models/"
    "cnn_5s_recording_loss_250train_locked17_seed42.pt",
    "muse_experiments/results/submissions/muse_cnn_recording_loss.zip",
    "muse_experiments/results/submissions/muse_cnn_250_recordings.zip",
    "muse_experiments/results/submissions/"
    "muse_weighted_lightgbm_pre_n2.zip",
    "muse_experiments/results/features/muse_features_100_pre_n2.csv",
    "muse_experiments/results/features/"
    "muse_features_250train_locked17_pre_n2.csv",
    "muse_experiments/results/tcn/windows_100_pre_n2.dat",
    "muse_experiments/results/tcn/windows_250train_locked17_pre_n2.dat",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes():
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in PROTECTED
    }


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss)
    return int(rss) * 1024


def _band(targets, prediction, low, high, closed):
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
        "windows": int(mask.sum()),
    }


def _pairs(frame):
    seen = []
    known = set()
    for subject, session in zip(frame["subject"], frame["session"]):
        key = (str(subject), str(session))
        if key not in known:
            known.add(key)
            seen.append(key)
    return seen


def _indices(frame, pairs):
    wanted = set(pairs)
    keys = list(zip(
        frame["subject"].astype(str),
        frame["session"].astype(str),
    ))
    return np.array(
        [index for index, key in enumerate(keys) if key in wanted],
        dtype=np.int64,
    )


def _result_path(fold, recordings):
    return OUTPUT_DIR / f"fold_{fold}_n{recordings}.json"


def _checkpoint_path(fold, recordings):
    return MODEL_DIR / f"cnn_fold_{fold}_n{recordings}.pt"


def report_capacity(inventory, assignment, locked):
    """Print fold sizes and refuse to train when 250 recordings are missing."""
    pairs = [(item["subject"], item["session"]) for item in inventory]
    subjects = {item["subject"] for item in inventory}
    windows = sum(item["windows"] for item in inventory)
    cache_gib = windows * 4 * 640 * 4 / (1024 ** 3)
    print("Eligible training subjects:", len(subjects))
    print("Eligible training recordings:", len(pairs))
    print("Eligible pre-N2 windows:", windows)
    print(f"Disk-backed window cache: {cache_gib:.2f} GiB")
    print("Locked holdout subjects excluded:", len(locked))
    short = []
    for fold in range(1, 6):
        held = {
            subject for subject, label in assignment.items() if label == fold
        }
        available = [pair for pair in pairs if pair[0] not in held]
        val_recordings = len(pairs) - len(available)
        print(
            f"Fold {fold}: validation subjects {len(held)}, "
            f"validation recordings {val_recordings}, "
            f"available training recordings {len(available)}"
        )
        if len(available) < 250:
            short.append(fold)
    if short:
        raise RuntimeError(
            "folds without 250 training recordings: "
            + ", ".join(str(fold) for fold in short)
        )
    print("Every fold can train both the 100- and 250-recording models.")
    return pairs


def _open_cache(path, rows):
    array = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(rows, 4, 640),
    )
    return WindowCache(path, array)


def _fit(frame, cache, train_idx, val_idx, cfg, fold, recordings):
    if len(recording_groups_safe(frame, train_idx)) != recordings:
        raise RuntimeError("training index does not match the recording count")
    set_seed(CV_SEED)
    choice = resolve_device(cfg.get("device"))
    if cfg.get("device") == "mps" and choice.name != "mps":
        raise RuntimeError("Apple MPS was requested but is not active")
    device = choice.device
    print("Torch device:", describe_device(choice))
    history = int(cfg["model"]["history_windows"])
    from muse_experiments.preprocessing.windows import SequenceDataset

    train_set = SequenceDataset(cache, frame, train_idx, history)
    val_set = SequenceDataset(cache, frame, val_idx, history)
    groups = recording_groups(train_set)
    if len(groups) != recordings:
        raise RuntimeError(
            f"grouped {len(groups)} recordings, expected {recordings}"
        )
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
            f"Fold {fold} n={recordings} epoch {epoch}: "
            f"train {train_loss:.2f} s, W-bMAE {score:.2f} s"
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            waited = 0
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
    mae, score, prediction, val_frame = best_result
    payload = _checkpoint_payload(
        model, cfg, choice, fold, recordings, best_epoch, mae, score
    )
    destination = _checkpoint_path(fold, recordings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(payload, destination)
    metrics = _metrics(
        fold, recordings, frame, train_idx, val_frame, prediction,
        best_epoch, elapsed, _rss_bytes(),
    )
    metrics["curves"] = curves
    metrics["checkpoint"] = str(destination.relative_to(REPO_ROOT))
    del model, optimizer, best_state
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return metrics, prediction, val_frame


def recording_groups_safe(frame, indices):
    """Recording keys in index order, used to catch a split recording."""
    subjects = frame["subject"].astype(str).to_numpy()
    sessions = frame["session"].astype(str).to_numpy()
    groups = []
    previous = None
    for index in indices:
        key = (subjects[int(index)], sessions[int(index)])
        if key != previous:
            groups.append(key)
            previous = key
    return groups


def _checkpoint_payload(
    model, cfg, choice, fold, recordings, epoch, mae, score
):
    model_cfg = cfg["model"]
    return {
        "state_dict": model.state_dict(),
        "config": "cnn_5s_recording_loss_cv",
        "fold": int(fold),
        "training_recordings": int(recordings),
        "cv_seed": CV_SEED,
        "history_windows": int(model_cfg["history_windows"]),
        "use_temporal": bool(model_cfg["use_temporal"]),
        "width": int(model_cfg["width"]),
        "kernel_size": int(model_cfg["kernel_size"]),
        "dilations": [int(item) for item in model_cfg["dilations"]],
        "dropout": float(model_cfg["dropout"]),
        "uv_scale": float(model_cfg["uv_scale"]),
        "channels": list(cfg["data"]["channels"]),
        "cap_seconds": float(cfg["evaluation"]["cap_seconds"]),
        "seed": CV_SEED,
        "epoch": int(epoch),
        "val_mae": float(mae),
        "val_wbmae": float(score),
        "device": choice.name,
        "loss": "recording_wbmae",
    }


def _metrics(
    fold, recordings, frame, train_idx, val_frame, prediction,
    best_epoch, elapsed, rss_bytes,
):
    targets = val_frame["target_s"].to_numpy()
    near = _band(targets, prediction, 0, 40, False)
    middle = _band(targets, prediction, 40, 90, False)
    approach = _band(targets, prediction, 90, 300, False)
    distant = _band(targets, prediction, 300, 600, True)
    train = frame.iloc[train_idx]
    return {
        "fold": int(fold),
        "training_recordings": int(recordings),
        "wbmae": float(weighted_bmae(val_frame, prediction, EDGES, WEIGHTS)),
        "mae": float(np.mean(np.abs(prediction - targets))),
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
        "train_subjects": int(train["subject"].nunique()),
        "train_windows": int(len(train_idx)),
        "val_subjects": int(val_frame["subject"].nunique()),
        "val_recordings": int(
            val_frame.groupby(["subject", "session"]).ngroups
        ),
        "best_epoch": int(best_epoch),
        "duration_s": float(elapsed),
        "peak_rss_bytes": int(rss_bytes),
    }


def _subject_rows(fold, left, right):
    rows = []
    subjects = sorted(set(left[1]["subject"].astype(str)))
    for subject in subjects:
        scores = {}
        for recordings, (prediction, frame) in ((100, left), (250, right)):
            mask = frame["subject"].astype(str).to_numpy() == subject
            scores[recordings] = weighted_bmae(
                frame.loc[mask], prediction[mask], EDGES, WEIGHTS
            )
        rows.append({
            "fold": fold,
            "subject": subject,
            "wbmae_100": scores[100],
            "wbmae_250": scores[250],
            "improvement_s": scores[100] - scores[250],
        })
    return rows


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _train_size(frame, cache, cfg, fold, pairs, val_subjects, recordings):
    path = _result_path(fold, recordings)
    if path.is_file():
        print(f"Fold {fold} n={recordings}: reusing {path.name}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved, None, None
    blocked = set(val_subjects)
    chosen = select_training_recordings(pairs, blocked, recordings)
    if recordings == 250:
        smaller = select_training_recordings(pairs, blocked, 100)
        if smaller != chosen[:100]:
            raise RuntimeError("the 100-recording set is not inside the 250")
    train_subjects = {subject for subject, _session in chosen}
    overlap = train_subjects & set(val_subjects)
    if overlap:
        raise RuntimeError(
            f"fold {fold} leaks subjects: {sorted(overlap)[:5]}"
        )
    train_idx = _indices(frame, chosen)
    val_idx = _indices(frame, [
        pair for pair in pairs if pair[0] in blocked
    ])
    assert_disjoint(
        frame.iloc[train_idx]["subject"],
        frame.iloc[val_idx]["subject"],
    )
    print(
        f"Fold {fold} n={recordings}: "
        f"train subjects {len(train_subjects)}, "
        f"train windows {len(train_idx)}, "
        f"validation windows {len(val_idx)}"
    )
    metrics, prediction, val_frame = _fit(
        frame, cache, train_idx, val_idx, cfg, fold, recordings
    )
    stored = dict(metrics)
    curves = stored.pop("curves")
    _write_json(path, stored | {"curves": curves})
    return stored | {"curves": curves}, prediction, val_frame


def _save_tables(assignment, inventory, results, subject_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for item in inventory:
        counts[item["subject"]] = counts.get(item["subject"], 0) + 1
    assignments = pd.DataFrame([
        {
            "subject": subject,
            "fold": fold,
            "recordings": counts[subject],
        }
        for subject, fold in sorted(assignment.items())
    ])
    assignments.to_csv(OUTPUT_DIR / "assignments.csv", index=False)
    flat = []
    curves = []
    for metrics in results:
        row = {key: value for key, value in metrics.items() if key != "curves"}
        flat.append(row)
        for point in metrics["curves"]:
            curves.append({
                "fold": metrics["fold"],
                "training_recordings": metrics["training_recordings"],
                **point,
            })
    pd.DataFrame(flat).to_csv(OUTPUT_DIR / "fold_scores.csv", index=False)
    pd.DataFrame(curves).to_csv(OUTPUT_DIR / "curves.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(
        OUTPUT_DIR / "subject_scores.csv", index=False
    )
    summary = _summary_table(results)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    _plot(summary, curves)
    return summary


def _summary_table(results):
    rows = []
    by_fold = {}
    for metrics in results:
        by_fold.setdefault(metrics["fold"], {})[
            metrics["training_recordings"]
        ] = metrics
    for fold in range(1, 6):
        small = by_fold[fold][100]["wbmae"]
        large = by_fold[fold][250]["wbmae"]
        rows.append({
            "fold": fold,
            "wbmae_100": small,
            "wbmae_250": large,
            "improvement_s": small - large,
        })
    frame = pd.DataFrame(rows)
    return frame


def _plot(summary, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = summary["fold"].astype(str)
    position = np.arange(len(summary))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        position - width / 2, summary["wbmae_100"], width,
        label="100 recordings",
    )
    axis.bar(
        position + width / 2, summary["wbmae_250"], width,
        label="250 recordings",
    )
    axis.set_xticks(position, folds)
    axis.set_xlabel("Validation fold")
    axis.set_ylabel("W-bMAE (seconds)")
    axis.set_title("Subject-disjoint validation W-bMAE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "wbmae_comparison.png", dpi=150)
    plt.close(figure)

    curve_frame = pd.DataFrame(curves)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for axis, column, label in (
        (axes[0], "train_loss", "Training loss (seconds)"),
        (axes[1], "val_wbmae", "Validation W-bMAE (seconds)"),
    ):
        series = (
            (100, "100 recordings"),
            (250, "250 recordings"),
        )
        for recordings, name in series:
            part = curve_frame[
                curve_frame["training_recordings"] == recordings
            ]
            color = "C0" if recordings == 100 else "C1"
            for _fold, group in part.groupby("fold"):
                axis.plot(
                    group["epoch"], group[column],
                    color=color, alpha=0.35, linewidth=1,
                )
            mean = part.groupby("epoch")[column].mean()
            axis.plot(
                mean.index, mean.to_numpy(),
                color=color, linewidth=2.0, label=name,
            )
        axis.set_ylabel(label)
        axis.legend()
    axes[1].set_xlabel("Epoch")
    axes[0].set_title("Training loss and validation W-bMAE")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150)
    plt.close(figure)


def _recommendation(results, subject_rows):
    by_size = {100: [], 250: []}
    for metrics in results:
        by_size[metrics["training_recordings"]].append(metrics)
    gains = []
    wins = 0
    for fold in range(1, 6):
        small = next(
            item["wbmae"] for item in by_size[100] if item["fold"] == fold
        )
        large = next(
            item["wbmae"] for item in by_size[250] if item["fold"] == fold
        )
        gains.append(small - large)
        if large < small:
            wins += 1
    large_epochs = [item["best_epoch"] for item in by_size[250]]
    at_cap = sum(epoch == 15 for epoch in large_epochs)
    subjects = pd.DataFrame(subject_rows)
    improved = int((subjects["improvement_s"] > 0).sum())
    return {
        "mean_100": float(np.mean(
            [item["wbmae"] for item in by_size[100]]
        )),
        "std_100": float(np.std(
            [item["wbmae"] for item in by_size[100]], ddof=1
        )),
        "mean_250": float(np.mean([item["wbmae"] for item in by_size[250]])),
        "std_250": float(np.std(
            [item["wbmae"] for item in by_size[250]], ddof=1
        )),
        "mean_improvement": float(np.mean(gains)),
        "std_improvement": float(np.std(gains, ddof=1)),
        "folds_won_by_250": int(wins),
        "subjects_improved": improved,
        "subjects": int(len(subjects)),
        "epochs_at_cap_250": int(at_cap),
        "best_epochs_250": large_epochs,
    }


def main():
    before = _protected_hashes()
    cfg = load_config("cnn_5s_recording_loss_250")
    root = repo_path(cfg["paths"]["muse_data_root"])
    locked = locked_holdout_subjects()
    inventory = inventory_training_pool(root, cfg, locked)
    assignment = assign_subject_folds(
        [item["subject"] for item in inventory],
        n_splits=5,
        seed=CV_SEED,
    )
    if set(assignment) & set(locked):
        raise RuntimeError("a locked subject entered cross-validation")
    pairs = report_capacity(inventory, assignment, locked)
    frame, cache_path, _locked = prepare_cv_pool()
    if set(frame["subject"].astype(str)) & set(locked):
        raise RuntimeError("the CV pool contains a locked subject")
    cache = _open_cache(cache_path, len(frame))
    pool_pairs = _pairs(frame)
    if set(pool_pairs) != set(pairs):
        raise RuntimeError("the feature table does not match the inventory")
    results = []
    subject_rows = []
    reusable = all(
        _result_path(fold, recordings).is_file()
        for fold in range(1, 6)
        for recordings in SIZES
    )
    for fold in range(1, 6):
        val_subjects = [
            subject for subject, label in assignment.items() if label == fold
        ]
        fold_outputs = []
        for recordings in SIZES:
            metrics, prediction, val_frame = _train_size(
                frame, cache, cfg, fold, pool_pairs,
                val_subjects, recordings,
            )
            results.append(metrics)
            fold_outputs.append((prediction, val_frame))
        if reusable:
            continue
        if fold_outputs[0][1] is None:
            continue
        left_frame = fold_outputs[0][1]
        right_frame = fold_outputs[1][1]
        if not np.array_equal(
            left_frame["target_s"].to_numpy(),
            right_frame["target_s"].to_numpy(),
        ):
            raise RuntimeError(
                "the two models saw different validation targets"
            )
        subject_rows.extend(_subject_rows(
            fold,
            (fold_outputs[0][0], left_frame),
            (fold_outputs[1][0], right_frame),
        ))
    if not subject_rows and (OUTPUT_DIR / "subject_scores.csv").is_file():
        subject_rows = pd.read_csv(
            OUTPUT_DIR / "subject_scores.csv"
        ).to_dict("records")
    summary = _save_tables(assignment, inventory, results, subject_rows)
    report = _recommendation(results, subject_rows)
    _write_json(OUTPUT_DIR / "recommendation.json", report)
    print(summary.to_string(index=False))
    print(
        "Mean W-bMAE 100: "
        f"{report['mean_100']:.2f} ± {report['std_100']:.2f} s"
    )
    print(
        "Mean W-bMAE 250: "
        f"{report['mean_250']:.2f} ± {report['std_250']:.2f} s"
    )
    print(
        "Mean improvement: "
        f"{report['mean_improvement']:.2f} ± "
        f"{report['std_improvement']:.2f} s"
    )
    print(
        f"Folds won by 250 recordings: {report['folds_won_by_250']} of 5"
    )
    print(
        "Validation subjects improved by 250 recordings: "
        f"{report['subjects_improved']} of {report['subjects']}"
    )
    print("250-recording best epochs:", report["best_epochs_250"])
    after = _protected_hashes()
    if after != before:
        raise RuntimeError("a protected research file changed during CV")
    print("Protected checkpoints and submissions were unchanged.")


if __name__ == "__main__":
    main()
