"""Preview one Muse recording without loading it all into memory."""

import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mne

from muse_experiments.config import load_config
from muse_experiments.io import repo_path


def main(argv=None):
    del argv
    cfg = load_config("default")
    root = repo_path(cfg["paths"]["muse_data_root"])
    files = sorted(root.rglob("*_eeg.vhdr"))
    if not files:
        raise FileNotFoundError("No Muse EEG recordings found.")

    vhdr = files[0]
    print("Recording:", vhdr.name)
    raw = mne.io.read_raw_brainvision(
        str(vhdr), preload=False, verbose="ERROR"
    )
    print("Channels:", raw.ch_names)
    print("Sampling rate:", raw.info["sfreq"], "Hz")

    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks="eeg", start=0, stop=int(5 * sfreq))
    print("EEG data shape:", data.shape)
    print("First five values:", data[0, :5] * 1e6, "uV")

    events_file = vhdr.with_name(
        vhdr.name.replace("_eeg.vhdr", "_events.tsv")
    )
    if not events_file.exists():
        print("Events file not found:", events_file)
        return

    with events_file.open(encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle, delimiter="\t"))
    for event in events:
        if event.get("trial_type") == "n2_onset":
            print("Events file:", events_file)
            print("Available columns:", list(event.keys()))
            print("N2 event:", event)
            return


if __name__ == "__main__":
    main()
