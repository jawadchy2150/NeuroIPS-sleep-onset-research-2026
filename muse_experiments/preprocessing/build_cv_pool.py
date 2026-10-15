"""Feature table for cross-validation, excluding the locked 17 subjects.

Recordings already stored for the 250-recording run are copied. New
official train sessions are read one at a time. Official test sessions
and the locked holdout are never written.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from muse_experiments.config import load_config
from muse_experiments.io import publish_file, repo_path
from muse_experiments.preprocessing.features import pre_n2_window_bounds
from muse_experiments.preprocessing.prepare_muse_features import (
    get_split,
    rows_for_recording,
)
from muse_experiments.preprocessing.windows import _meta_path, _source_stamp
from muse_experiments.splits import _label_key
from muse_experiments.training.lightgbm_run import split_subjects

POOL_CSV = (
    "muse_experiments/results/features/"
    "muse_features_cv_pool_locked17_pre_n2.csv"
)
POOL_CACHE = (
    "muse_experiments/results/tcn/windows_cv_pool_locked17_pre_n2.dat"
)


def locked_holdout_subjects():
    """The original 17 validation subjects, recovered from their table."""
    cfg = load_config("lightgbm_weighted")
    frame = pd.read_csv(repo_path(cfg["inputs"]["features_csv"]))
    _train_idx, val_idx = split_subjects(frame, cfg)
    found = sorted(frame.iloc[val_idx]["subject"].astype(str).unique())
    listed = load_config("cnn_5s_recording_loss_250")
    listed = sorted(listed["split"]["validation_subjects"])
    if found != [str(subject) for subject in listed]:
        raise RuntimeError("locked holdout does not match the saved split")
    return found


def _sessions(root, subject, cache):
    if subject not in cache:
        path = Path(root) / subject / f"{subject}_sessions.tsv"
        with path.open(encoding="utf-8-sig") as handle:
            cache[subject] = list(csv.DictReader(handle, delimiter="\t"))
    return cache[subject]


def _session_row(root, subject, session, cache):
    for row in _sessions(root, subject, cache):
        if row["session_id"] == session:
            return row
    raise ValueError(f"missing session metadata: {subject}/{session}")


def expected_windows(meta, sampling_hz, window_samples):
    bounds = pre_n2_window_bounds(
        int(float(meta["sample_count"])),
        float(sampling_hz),
        int(window_samples),
        float(meta["n2_onset"]),
    )
    return len(bounds)


def inventory_training_pool(root, cfg, locked):
    """Official train recordings whose subjects are outside the holdout."""
    locked = set(locked)
    session_cache = {}
    rows = []
    for header in sorted(Path(root).rglob("*_eeg.vhdr")):
        subject = header.parents[2].name
        session = header.parents[1].name
        meta = _session_row(root, subject, session, session_cache)
        split = meta["split"].strip().lower()
        if split != cfg["split"]["train_value"]:
            continue
        if subject in locked:
            continue
        if get_split(root, subject, session) != split:
            raise RuntimeError(f"{subject}/{session} split metadata disagrees")
        windows = expected_windows(
            meta,
            cfg["data"]["sampling_hz"],
            cfg["data"]["window_samples"],
        )
        if windows < 1:
            raise RuntimeError(f"{subject}/{session} has no pre-N2 window")
        rows.append({
            "subject": subject,
            "session": session,
            "path": header,
            "windows": windows,
        })
    rows.sort(
        key=lambda item: (
            _label_key(item["subject"]) + _label_key(item["session"])
        )
    )
    return rows


def _old_groups(frame, locked):
    grouped = {}
    for index, row in frame.iterrows():
        subject = str(row["subject"])
        if subject in locked:
            continue
        key = (subject, str(row["session"]))
        grouped.setdefault(key, []).append(int(index))
    return grouped


def _write_pool(plan, cfg, old_cache, destination_csv, destination_cache):
    sample = next(item for item in plan if item["kind"] == "copy")
    columns = list(sample["rows"].columns)
    channels = list(cfg["data"]["channels"])
    samples = int(cfg["data"]["window_samples"])
    total = sum(item["windows"] for item in plan)
    sfreq = float(cfg["data"]["sampling_hz"])

    def write_csv(path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in plan:
                if item["kind"] == "copy":
                    for _, row in item["rows"].iterrows():
                        writer.writerow(row.to_dict())
                else:
                    for row in item["extracted"]:
                        writer.writerow(row)

    def write_cache(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = np.memmap(
            path,
            dtype=np.float32,
            mode="w+",
            shape=(total, len(channels), samples),
        )
        import mne

        cursor = 0
        source = np.memmap(
            old_cache,
            dtype=np.float32,
            mode="r",
            shape=(
                int(cfg["_old_cache_rows"]),
                len(channels),
                samples,
            ),
        )
        for item in plan:
            count = item["windows"]
            if item["kind"] == "copy":
                indices = item["indices"]
                for offset, index in enumerate(indices):
                    cache[cursor + offset] = source[index]
                if not np.array_equal(
                    cache[cursor:cursor + count],
                    np.stack([source[index] for index in indices]),
                ):
                    raise RuntimeError(
                        f"copied EEG mismatch for {item['subject']}"
                    )
            else:
                raw = mne.io.read_raw_brainvision(
                    str(item["path"]),
                    preload=False,
                    verbose="ERROR",
                )
                try:
                    for offset, row in enumerate(item["extracted"]):
                        start = int(round(
                            float(row["window_start_s"]) * sfreq
                        ))
                        stop = int(round(float(row["window_end_s"]) * sfreq))
                        data = raw.get_data(
                            picks=channels, start=start, stop=stop
                        )
                        cache[cursor + offset] = np.asarray(
                            data * 1e6, dtype=np.float32
                        )
                finally:
                    raw.close()
            cursor += count
            print(
                f"{item['subject']}/{item['session']}: "
                f"stored {count} windows"
            )
        if cursor != total:
            raise RuntimeError(f"stored {cursor} windows, planned {total}")
        cache.flush()
        del cache
        del source

    csv_path = publish_file(destination_csv, write_csv)
    cache_path = publish_file(destination_cache, write_cache)
    stamp = _source_stamp(csv_path)
    stamp.update({
        "rows": int(total),
        "channels": channels,
        "samples": samples,
    })
    _meta_path(cache_path).write_text(json.dumps(stamp), encoding="utf-8")
    print("Saved CV pool:", csv_path)
    print("Saved CV cache:", cache_path)
    return csv_path, cache_path


def prepare_cv_pool():
    """Build the holdout-free pool, or reuse the saved files."""
    cfg = load_config("cnn_5s_recording_loss_250")
    root = repo_path(cfg["paths"]["muse_data_root"])
    locked = set(locked_holdout_subjects())
    csv_path = repo_path(POOL_CSV)
    cache_path = repo_path(POOL_CACHE)
    if csv_path.is_file() and cache_path.is_file():
        frame = pd.read_csv(csv_path)
        print(
            "Reusing CV pool:",
            frame.groupby(["subject", "session"]).ngroups,
            "recordings,",
            frame["subject"].nunique(),
            "subjects,",
            len(frame),
            "windows",
        )
        return frame, cache_path, sorted(locked)

    inventory = inventory_training_pool(root, cfg, locked)
    old_cfg = load_config("cnn_5s_recording_loss_250")
    old_frame = pd.read_csv(repo_path(old_cfg["inputs"]["features_csv"]))
    grouped = _old_groups(old_frame, locked)
    old_cache = repo_path(old_cfg["outputs"]["window_cache"])
    plan = []
    for item in inventory:
        key = (item["subject"], item["session"])
        if key in grouped:
            indices = grouped[key]
            rows = old_frame.loc[indices].reset_index(drop=True)
            if len(rows) != item["windows"]:
                raise RuntimeError(
                    f"{key[0]}/{key[1]} copied windows {len(rows)} "
                    f"!= metadata {item['windows']}"
                )
            plan.append({
                "kind": "copy",
                "subject": key[0],
                "session": key[1],
                "windows": len(rows),
                "indices": indices,
                "rows": rows,
            })
        else:
            extracted = rows_for_recording(root, cfg, item["path"])
            if len(extracted) != item["windows"]:
                raise RuntimeError(
                    f"{key[0]}/{key[1]} extracted {len(extracted)} "
                    f"windows, metadata says {item['windows']}"
                )
            plan.append({
                "kind": "extract",
                "subject": key[0],
                "session": key[1],
                "path": item["path"],
                "windows": len(extracted),
                "extracted": extracted,
            })
    copied = sum(item["kind"] == "copy" for item in plan)
    print(
        f"CV pool plan: {len(plan)} recordings, "
        f"{copied} copied, {len(plan) - copied} read from disk"
    )
    cfg = dict(cfg)
    cfg["_old_cache_rows"] = len(old_frame)
    _write_pool(plan, cfg, old_cache, csv_path, cache_path)
    frame = pd.read_csv(csv_path)
    return frame, cache_path, sorted(locked)
