"""Add official training recordings without moving the 17-subject holdout.

The original feature table is copied first, so its 83 training recordings
and 17 validation recordings stay byte-for-byte. Extra rows come from
other official train sessions whose subjects are not in that holdout.
"""

import csv
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from muse_experiments.config import load_config
from muse_experiments.io import publish_file, repo_path
from muse_experiments.preprocessing.features import (
    METADATA_COLUMNS,
    feature_columns,
)
from muse_experiments.preprocessing.prepare_muse_features import (
    get_n2_onset,
    get_split,
    rows_for_recording,
    subject_has_official_test,
)
from muse_experiments.preprocessing.windows import _meta_path, _source_stamp
from muse_experiments.splits import assert_disjoint
from muse_experiments.training.lightgbm_run import split_subjects

TARGET_TRAIN = 250
OLD_CONFIG = "lightgbm_weighted"
NEW_CONFIG = "cnn_5s_recording_loss_250"


def recording_pairs(frame):
    """(subject, session) in first-seen order."""
    pairs = []
    seen = set()
    for subject, session in zip(frame["subject"], frame["session"]):
        key = (str(subject), str(session))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def choose_training_keys(sorted_keys, locked_keys, blocked, target):
    """Keep ``locked_keys`` and then add other subjects up to ``target``."""
    blocked = {str(subject) for subject in blocked}
    chosen = [(str(subject), str(session)) for subject, session in locked_keys]
    for subject, _session in chosen:
        if subject in blocked:
            raise ValueError(
                f"a retained training recording belongs to {subject}"
            )
    seen = set(chosen)
    for subject, session in sorted_keys:
        if len(chosen) >= target:
            break
        key = (str(subject), str(session))
        if key[0] in blocked or key in seen:
            continue
        chosen.append(key)
        seen.add(key)
    if len(chosen) < target:
        raise ValueError(
            f"only {len(chosen)} eligible training recordings, "
            f"need {target}"
        )
    return chosen[:target]


def _header(vhdr_root, subject, session):
    folder = Path(vhdr_root) / subject / session
    matches = sorted(folder.rglob("*_eeg.vhdr"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one header for {subject}/{session}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _candidate_headers(root, cfg, locked, blocked):
    train_value = cfg["split"]["train_value"]
    chosen = []
    for vhdr in sorted(Path(root).rglob("*_eeg.vhdr")):
        subject = vhdr.parents[2].name
        session = vhdr.parents[1].name
        if get_split(root, subject, session) != train_value:
            continue
        if subject in blocked or (subject, session) in locked:
            continue
        chosen.append(vhdr)
    return chosen


def _original_split(old_frame, old_cfg):
    train_idx, val_idx = split_subjects(old_frame, old_cfg)
    train = old_frame.iloc[train_idx]
    val = old_frame.iloc[val_idx]
    return recording_pairs(train), recording_pairs(val)


def _write_table(old_csv, destination, root, cfg, candidates):
    columns = METADATA_COLUMNS + feature_columns(
        list(cfg["data"]["channels"]), cfg["data"]["bands"]
    )
    added = []
    skipped_empty = 0

    def write_temp(path):
        nonlocal skipped_empty
        payload = Path(old_csv).read_bytes()
        if not payload.endswith(b"\n"):
            payload += b"\n"
        with path.open("wb") as handle:
            handle.write(payload)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            for vhdr in candidates:
                if len(added) >= TARGET_TRAIN - 83:
                    break
                rows = rows_for_recording(root, cfg, vhdr)
                if not rows:
                    skipped_empty += 1
                    print(
                        f"{vhdr.parents[2].name}/{vhdr.parents[1].name}: "
                        "skipped, no pre-N2 window"
                    )
                    continue
                for row in rows:
                    writer.writerow(row)
                added.append((
                    rows[0]["subject"],
                    rows[0]["session"],
                    len(rows),
                ))
                print(
                    f"{rows[0]['subject']}/{rows[0]['session']}: "
                    f"{len(rows)} windows added"
                )

    written = publish_file(destination, write_temp)
    print(f"Skipped empty recordings: {skipped_empty}")
    print("Saved feature table:", written)
    return written, added


def _copy_prefix(old_cache, new_cache, n_old, channels, samples):
    source = np.memmap(
        old_cache,
        dtype=np.float32,
        mode="r",
        shape=(n_old, channels, samples),
    )
    for start in range(0, n_old, 512):
        stop = min(start + 512, n_old)
        new_cache[start:stop] = source[start:stop]
        if not np.array_equal(new_cache[start:stop], source[start:stop]):
            raise RuntimeError("copied EEG windows do not match the cache")
    del source


def _write_cache(frame, root, cfg, old_cache, destination, n_old):
    channels = list(cfg["data"]["channels"])
    samples = int(cfg["data"]["window_samples"])
    sfreq = float(cfg["data"]["sampling_hz"])

    def write_temp(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = np.memmap(
            path,
            dtype=np.float32,
            mode="w+",
            shape=(len(frame), len(channels), samples),
        )
        _copy_prefix(old_cache, cache, n_old, len(channels), samples)
        row = n_old
        fresh = frame.iloc[n_old:]
        grouped = fresh.groupby(["subject", "session"], sort=False)
        for (subject, session), part in grouped:
            if int(part.index.min()) != row:
                raise ValueError(
                    f"{subject}/{session} is not contiguous at row {row}"
                )
            header = _header(root, subject, session)
            raw = mne.io.read_raw_brainvision(
                str(header), preload=False, verbose="ERROR"
            )
            try:
                for _, window in part.iterrows():
                    start = int(round(float(window["window_start_s"]) * sfreq))
                    stop = int(round(float(window["window_end_s"]) * sfreq))
                    data = raw.get_data(
                        picks=channels, start=start, stop=stop
                    )
                    cache[row] = np.asarray(data * 1e6, dtype=np.float32)
                    row += 1
            finally:
                raw.close()
            print(f"{subject}/{session}: cached {len(part)} windows")
        if row != len(frame):
            raise ValueError(
                f"cached {row} windows, table has {len(frame)}"
            )
        cache.flush()
        del cache

    written = publish_file(destination, write_temp)
    stamp = _source_stamp(repo_path(cfg["inputs"]["features_csv"]))
    stamp.update({
        "rows": int(len(frame)),
        "channels": channels,
        "samples": samples,
    })
    _meta_path(written).write_text(json.dumps(stamp), encoding="utf-8")
    print("Saved window cache:", written)
    return written


def _check_targets(frame, root):
    onsets = {}
    for subject, session in recording_pairs(frame):
        onsets[(subject, session)] = get_n2_onset(
            _header(root, subject, session)
        )
    previous = {}
    for row in frame.itertuples(index=False):
        key = (str(row.subject), str(row.session))
        onset = onsets[key]
        start = float(row.window_start_s)
        end = float(row.window_end_s)
        if abs((end - start) - 5.0) > 1e-9:
            raise RuntimeError(f"{key} window is not 5 seconds")
        if end > onset + 1e-9:
            raise RuntimeError(f"{key} window ends after N2")
        expected = float(np.clip(onset - end, 0.0, 600.0))
        if abs(float(row.target_s) - expected) > 1e-4:
            raise RuntimeError(f"{key} target does not match N2 onset")
        last = previous.get(key)
        if last is None:
            if abs(start) > 1e-9:
                raise RuntimeError(f"{key} does not start at 0")
        elif abs(start - (last + 5.0)) > 1e-6:
            raise RuntimeError(f"{key} windows are not contiguous")
        previous[key] = start


def _check_one_recording(frame, root, cfg, subject, session):
    header = _header(root, subject, session)
    fresh = rows_for_recording(root, cfg, header)
    saved = frame[
        (frame["subject"] == subject) & (frame["session"] == session)
    ]
    if len(fresh) != len(saved):
        raise RuntimeError(
            f"{subject}/{session} re-extract has {len(fresh)} windows, "
            f"table has {len(saved)}"
        )
    columns = feature_columns(
        list(cfg["data"]["channels"]), cfg["data"]["bands"]
    )
    for column in columns + ["target_s", "window_start_s", "window_end_s"]:
        left = saved[column].to_numpy(dtype=np.float64)
        right = np.array([float(row[column]) for row in fresh])
        if not np.allclose(left, right, rtol=0, atol=1e-4):
            raise RuntimeError(
                f"{subject}/{session} {column} does not match re-extraction"
            )


def _check_cache_sample(frame, cache_path, root, cfg, n_old):
    channels = list(cfg["data"]["channels"])
    samples = int(cfg["data"]["window_samples"])
    sfreq = float(cfg["data"]["sampling_hz"])
    cache = np.memmap(
        cache_path,
        dtype=np.float32,
        mode="r",
        shape=(len(frame), len(channels), samples),
    )
    old_cache = repo_path(
        load_config("tcn_120s")["outputs"]["window_cache"]
    )
    old = np.memmap(
        old_cache,
        dtype=np.float32,
        mode="r",
        shape=(n_old, len(channels), samples),
    )
    for start in range(0, n_old, 512):
        stop = min(start + 512, n_old)
        if not np.array_equal(cache[start:stop], old[start:stop]):
            raise RuntimeError(
                "validation EEG no longer matches the old cache"
            )
    del old
    pairs = recording_pairs(frame.iloc[n_old:])
    for subject, session in pairs[:2]:
        part = frame[
            (frame["subject"] == subject) & (frame["session"] == session)
        ]
        raw = mne.io.read_raw_brainvision(
            str(_header(root, subject, session)),
            preload=False,
            verbose="ERROR",
        )
        try:
            for index, window in part.iterrows():
                start = int(round(float(window["window_start_s"]) * sfreq))
                stop = int(round(float(window["window_end_s"]) * sfreq))
                data = raw.get_data(picks=channels, start=start, stop=stop)
                stored = np.asarray(data * 1e6, dtype=np.float32)
                if not np.array_equal(cache[index], stored):
                    raise RuntimeError(
                        f"{subject}/{session} cache is not microvolts"
                    )
        finally:
            raw.close()
    del cache


def validate_expanded_table(frame, root, cfg, old_frame, n_old):
    """Raise before training if the holdout or the window rules moved."""
    old_cfg = load_config(OLD_CONFIG)
    locked_train, locked_val = _original_split(old_frame, old_cfg)
    if len(locked_train) != 83 or len(locked_val) != 17:
        raise RuntimeError("the original table is not the 83/17 split")
    listed = [str(item) for item in cfg["split"]["validation_subjects"]]
    if sorted(listed) != sorted(subject for subject, _session in locked_val):
        raise RuntimeError(
            "config validation subjects differ from the holdout"
        )
    prefix = frame.iloc[:n_old].reset_index(drop=True)
    reference = old_frame.reset_index(drop=True)
    if not prefix.equals(reference):
        raise RuntimeError("the original feature rows were not copied exactly")

    train_idx, val_idx = split_subjects(frame, cfg)
    train = frame.iloc[train_idx]
    val = frame.iloc[val_idx]
    train_pairs = recording_pairs(train)
    val_pairs = recording_pairs(val)
    if val_pairs != locked_val:
        raise RuntimeError("validation recordings are not the original 17")
    if len(train_pairs) != TARGET_TRAIN:
        raise RuntimeError(
            f"training recordings are {len(train_pairs)}, not {TARGET_TRAIN}"
        )
    if not set(locked_train).issubset(train_pairs):
        raise RuntimeError("an original training recording was dropped")
    overlap = set(train["subject"]).intersection(val["subject"])
    if overlap:
        raise RuntimeError(f"subject leakage: {sorted(overlap)[:5]}")
    assert_disjoint(train["subject"], val["subject"])
    blocked = {subject for subject, _session in locked_val}
    extra_val = [
        pair for pair in recording_pairs(frame)
        if pair[0] in blocked and pair not in locked_val
    ]
    if extra_val:
        raise RuntimeError(
            f"validation subjects contributed extra sessions: {extra_val[:3]}"
        )

    test_value = cfg["split"]["test_value"]
    train_value = cfg["split"]["train_value"]
    for subject, session in recording_pairs(frame):
        split = get_split(root, subject, session)
        if split != train_value:
            raise RuntimeError(f"{subject}/{session} split is {split}")
        flag = int(subject_has_official_test(root, subject, test_value))
        saved = frame.loc[
            (frame["subject"] == subject) & (frame["session"] == session),
            "has_official_test",
        ]
        if not np.all(saved.to_numpy() == flag):
            raise RuntimeError(f"{subject} has_official_test is inconsistent")
    _check_targets(frame, root)
    _check_one_recording(
        frame, root, cfg, locked_train[0][0], locked_train[0][1]
    )
    new_pairs = recording_pairs(frame.iloc[n_old:])
    if new_pairs:
        _check_one_recording(
            frame, root, cfg, new_pairs[0][0], new_pairs[0][1]
        )
    cache_path = repo_path(cfg["outputs"]["window_cache"])
    _check_cache_sample(frame, cache_path, root, cfg, n_old)

    summary = {
        "train_recordings": len(train_pairs),
        "train_subjects": int(train["subject"].nunique()),
        "train_windows": int(len(train)),
        "val_recordings": len(val_pairs),
        "val_subjects": int(val["subject"].nunique()),
        "val_windows": int(len(val)),
        "added_recordings": len(new_pairs),
        "added_from_original_subjects": sum(
            subject in {item[0] for item in locked_train}
            for subject, _session in new_pairs
        ),
    }
    print("Expanded dataset checks passed")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def prepare_expanded_table():
    """Build the 250-recording table when needed, then validate it."""
    old_cfg = load_config(OLD_CONFIG)
    cfg = load_config(NEW_CONFIG)
    root = repo_path(cfg["paths"]["muse_data_root"])
    old_csv = repo_path(old_cfg["inputs"]["features_csv"])
    old_frame = pd.read_csv(old_csv)
    locked_train, locked_val = _original_split(old_frame, old_cfg)
    blocked = {subject for subject, _session in locked_val}
    desired_csv = repo_path(cfg["outputs"]["features_csv"])
    desired_cache = repo_path(cfg["outputs"]["window_cache"])
    if not desired_csv.exists():
        candidates = _candidate_headers(
            root, cfg, set(locked_train), blocked
        )
        _write_table(old_csv, desired_csv, root, cfg, candidates)
    frame = pd.read_csv(desired_csv)
    if not desired_cache.exists():
        old_cache = repo_path(
            load_config("tcn_120s")["outputs"]["window_cache"]
        )
        _write_cache(
            frame, root, cfg, old_cache, desired_cache, len(old_frame)
        )
    return validate_expanded_table(frame, root, cfg, old_frame, len(old_frame))
