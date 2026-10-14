"""Build the Muse feature table from official train recordings."""

import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mne
import numpy as np

from muse_experiments.config import load_config
from muse_experiments.io import publish_file, repo_path
from muse_experiments.preprocessing.features import (
    METADATA_COLUMNS,
    extract_features,
    feature_columns,
)


def get_split(root, subject, session):
    """Read the official train/test assignment."""
    path = root / subject / f"{subject}_sessions.tsv"
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["session_id"] == session:
                return row["split"].strip().lower()
    return None


def get_n2_onset(vhdr):
    """Read N2 onset from the annotation file."""
    path = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
    with path.open(encoding="utf-8-sig") as handle:
        for event in csv.DictReader(handle, delimiter="\t"):
            if event["trial_type"] == "n2_onset":
                return float(event["onset"])
    raise ValueError(f"N2 onset not found: {vhdr}")


def iter_feature_rows(root, cfg, stats):
    """Yield one feature row per 5-second train window."""
    channels = list(cfg["data"]["channels"])
    bands = cfg["data"]["bands"]
    window_seconds = cfg["data"]["window_seconds"]
    cap_seconds = cfg["data"]["cap_seconds"]
    max_recordings = cfg["data"]["max_recordings"]
    train_value = cfg["split"]["train_value"]
    files = sorted(Path(root).rglob("*_eeg.vhdr"))
    processed_subjects = set()

    for vhdr in files:
        if len(processed_subjects) >= max_recordings:
            break
        subject = vhdr.parents[2].name
        session = vhdr.parents[1].name
        if subject in processed_subjects:
            continue
        if get_split(root, subject, session) != train_value:
            continue

        n2_onset = get_n2_onset(vhdr)
        raw = mne.io.read_raw_brainvision(
            str(vhdr),
            preload=False,
            verbose="ERROR",
        )
        if not all(channel in raw.ch_names for channel in channels):
            raw.close()
            continue

        sfreq = raw.info["sfreq"]
        n_samples = int(window_seconds * sfreq)
        count = 0
        for start in range(0, raw.n_times - n_samples + 1, n_samples):
            end = start + n_samples
            data = raw.get_data(picks=channels, start=start, stop=end)
            window_end_s = end / sfreq
            target = float(np.clip(n2_onset - window_end_s, 0, cap_seconds))
            row = {
                "subject": subject,
                "session": session,
                "window_start_s": start / sfreq,
                "window_end_s": window_end_s,
                "target_s": target,
            }
            row.update(extract_features(data, sfreq, channels, bands))
            count += 1
            yield row

        raw.close()
        processed_subjects.add(subject)
        stats["recordings"] += 1
        print(f"{subject}/{session}: {count} windows prepared")


def write_feature_table(root, cfg, desired=None):
    """Write a new feature CSV. An existing file is left untouched."""
    channels = list(cfg["data"]["channels"])
    columns = METADATA_COLUMNS + feature_columns(
        channels, cfg["data"]["bands"]
    )
    if desired is None:
        desired = repo_path(cfg["outputs"]["features_csv"])
    stats = {"recordings": 0, "windows": 0}

    def write_temp(temporary):
        stats["recordings"] = 0
        stats["windows"] = 0
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in iter_feature_rows(root, cfg, stats):
                writer.writerow(row)
                stats["windows"] += 1

    path = publish_file(desired, write_temp)
    print("\nPreparation completed!")
    print("Recordings:", stats["recordings"])
    print("Total windows:", stats["windows"])
    print("Saved CSV:", path)
    return path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    name = args[0] if args else "preprocess_muse_100"
    cfg = load_config(name)
    root = repo_path(cfg["paths"]["muse_data_root"])
    write_feature_table(root, cfg)


if __name__ == "__main__":
    main()
