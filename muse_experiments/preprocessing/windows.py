"""Raw Muse windows for the temporal model.

The corrected feature table supplies the window list, targets, and the
subject split. EEG itself is read one recording at a time into a
memory-mapped array. A training item then views at most ``history``
windows from that same recording.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from muse_experiments.io import allocate_output, repo_path


def recording_starts(subjects, sessions):
    """First row of the contiguous recording that contains each row."""
    subjects = np.asarray(subjects)
    sessions = np.asarray(sessions)
    if len(subjects) != len(sessions):
        raise ValueError("subjects and sessions must have the same length")
    starts = np.empty(len(subjects), dtype=np.int64)
    current = 0
    previous = None
    for index, key in enumerate(zip(subjects.tolist(), sessions.tolist())):
        if key != previous:
            current = index
            previous = key
        starts[index] = current
    return starts


def history_bounds(record_start, row, history):
    """Inclusive sample range for one causal history.

    The range never starts before the recording and never passes ``row``.
    """
    if history < 1:
        raise ValueError("history must be at least 1")
    if row < record_start:
        raise ValueError("row is before its recording start")
    first = max(int(record_start), int(row) - int(history) + 1)
    return first, int(row)


def stack_history(windows, history):
    """Left-pad a list of ``(C, samples)`` windows. Returns array and mask."""
    if not windows:
        raise ValueError("windows is empty")
    channels, samples = windows[0].shape
    sequence = np.zeros((history, channels, samples), dtype=np.float32)
    mask = np.zeros(history, dtype=np.float32)
    kept = windows[-history:]
    sequence[-len(kept):] = np.stack(kept).astype(np.float32)
    mask[-len(kept):] = 1.0
    return sequence, mask


class WindowCache:
    """Memory-mapped microvolt windows aligned with a feature table."""

    def __init__(self, path, array):
        self.path = Path(path)
        self.array = array

    def __len__(self):
        return int(self.array.shape[0])

    @property
    def shape(self):
        return tuple(self.array.shape)


def _meta_path(path):
    return Path(path).with_suffix(".json")


def _source_stamp(csv_path):
    stat = Path(csv_path).stat()
    return {
        "features_csv": str(csv_path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _open_cache(path, rows, channels, samples):
    array = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        shape=(rows, channels, samples),
    )
    return WindowCache(path, array)


def ensure_window_cache(frame, cfg, root):
    """Return a cache for ``frame``, building it one recording at a time."""
    desired = repo_path(cfg["outputs"]["window_cache"])
    channels = list(cfg["data"]["channels"])
    samples = int(cfg["data"]["window_samples"])
    stamp = _source_stamp(repo_path(cfg["inputs"]["features_csv"]))
    stamp.update({
        "rows": int(len(frame)),
        "channels": channels,
        "samples": samples,
    })
    meta_path = _meta_path(desired)
    if desired.is_file() and meta_path.is_file():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == stamp:
            return _open_cache(desired, len(frame), len(channels), samples)
    destination = allocate_output(desired)
    _write_cache(frame, root, destination, channels, samples, cfg)
    _meta_path(destination).write_text(
        json.dumps(stamp),
        encoding="utf-8",
    )
    print("Saved window cache:", destination)
    return _open_cache(destination, len(frame), len(channels), samples)


def _recording_path(root, subject, session):
    folder = Path(root) / subject / session
    matches = sorted(folder.rglob("*_eeg.vhdr"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one BrainVision header for {subject}/{session}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _write_cache(frame, root, destination, channels, samples, cfg):
    import mne

    sfreq = float(cfg["data"]["sampling_hz"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache = np.memmap(
        destination,
        dtype=np.float32,
        mode="w+",
        shape=(len(frame), len(channels), samples),
    )
    grouped = frame.groupby(["subject", "session"], sort=False)
    row = 0
    for (subject, session), part in grouped:
        if int(part.index.min()) != row:
            raise ValueError(
                f"{subject}/{session} is not contiguous in the feature table"
            )
        header = _recording_path(root, subject, session)
        raw = mne.io.read_raw_brainvision(
            str(header),
            preload=False,
            verbose="ERROR",
        )
        try:
            if abs(float(raw.info["sfreq"]) - sfreq) > 1e-6:
                raise ValueError(
                    f"{subject}/{session} is {raw.info['sfreq']} Hz"
                )
            for _, window in part.iterrows():
                start = int(round(float(window["window_start_s"]) * sfreq))
                stop = int(round(float(window["window_end_s"]) * sfreq))
                if stop - start != samples:
                    raise ValueError(
                        f"{subject}/{session} window is not {samples} samples"
                    )
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
            f"cached {row} windows, feature table has {len(frame)}"
        )
    cache.flush()


class SequenceDataset(Dataset):
    """Causal histories drawn from one recording's cached windows."""

    def __init__(self, cache, frame, indices, history):
        self.cache = cache.array
        self.targets = frame["target_s"].to_numpy(dtype=np.float32)
        self.subjects = frame["subject"].astype(str).to_numpy()
        self.sessions = frame["session"].astype(str).to_numpy()
        self.starts = recording_starts(self.subjects, self.sessions)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.history = int(history)
        if self.cache.shape[0] != len(frame):
            raise ValueError("cache rows do not match the feature table")

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, index):
        row = int(self.indices[index])
        first, last = history_bounds(
            self.starts[row], row, self.history
        )
        sequence, mask = stack_history(
            [self.cache[item] for item in range(first, last + 1)],
            self.history,
        )
        return (
            torch.from_numpy(sequence),
            torch.from_numpy(mask),
            torch.tensor(self.targets[row]),
            self.subjects[row],
            self.sessions[row],
        )


def load_feature_table(cfg):
    path = repo_path(cfg["inputs"]["features_csv"])
    if not path.is_file():
        raise FileNotFoundError(f"feature table not found: {path}")
    frame = pd.read_csv(path).reset_index(drop=True)
    required = {
        "subject",
        "session",
        "window_start_s",
        "window_end_s",
        "target_s",
        "has_official_test",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature table is missing {sorted(missing)}")
    return frame
