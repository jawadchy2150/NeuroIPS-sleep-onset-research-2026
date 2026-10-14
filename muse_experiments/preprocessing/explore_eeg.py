"""Preview one Sleep-EDF recording without loading it all into memory."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from muse_experiments.config import load_config
from muse_experiments.io import REPO_ROOT, publish_file, repo_path


def save_figure(figure, desired, dpi=None):
    def write_temp(temporary):
        if dpi is None:
            figure.savefig(temporary)
        else:
            figure.savefig(temporary, dpi=dpi)
        plt.close(figure)

    return publish_file(desired, write_temp)


def main(argv=None):
    del argv
    cfg = load_config("default")
    data_dir = REPO_ROOT / "tracks" / "sleep_onset" / "data"
    figures = repo_path(cfg["outputs"]["figures_dir"])
    files = sorted(data_dir.rglob("*-PSG.edf"))
    if not files:
        raise FileNotFoundError("No PSG EDF files found in the data folder.")

    psg_file = files[0]
    print("Recording:", psg_file.name)
    raw = mne.io.read_raw_edf(psg_file, preload=False, verbose="ERROR")
    print("Channels:", raw.ch_names)
    print("Sampling frequency:", raw.info["sfreq"], "Hz")

    eeg_channels = [
        channel for channel in raw.ch_names if channel.startswith("EEG")
    ]
    channel = eeg_channels[0]
    sfreq = raw.info["sfreq"]
    duration = 10
    signal = raw.get_data(
        picks=[channel],
        start=0,
        stop=int(sfreq * duration),
    )[0] * 1e6

    print("Selected channel:", channel)
    print("First 10 numerical values:", signal[:10])
    print("Number of samples:", len(signal))

    times = np.arange(len(signal)) / sfreq
    figure = plt.figure(figsize=(12, 4))
    plt.plot(times, signal, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (microvolts)")
    plt.title(f"Real EEG Recording: {channel}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    preview = save_figure(figure, figures / "eeg_preview.png", dpi=150)
    print("Saved figure:", preview)

    hypnograms = sorted(
        psg_file.parent.glob(psg_file.name[:6] + "*Hypnogram.edf")
    )
    if not hypnograms:
        print("Matching hypnogram not found.")
        return

    annotations = mne.read_annotations(hypnograms[0])
    print("\nSleep-stage annotations:")
    for onset, stage_duration, stage in zip(
        annotations.onset[:10],
        annotations.duration[:10],
        annotations.description[:10],
    ):
        print(f"{onset:.0f}s | {stage_duration:.0f}s | {stage}")

    n2_times = annotations.onset[annotations.description == "Sleep stage 2"]
    if len(n2_times) == 0:
        print("Matching hypnogram not found.")
        return

    print(f"\nFirst N2 starts at: {n2_times[0]:.0f} seconds")
    first_n2 = float(n2_times[0])
    window_end = first_n2 - 60
    window_start = window_end - 5
    eeg_window = raw.get_data(
        picks=[channel],
        start=int(window_start * sfreq),
        stop=int(window_end * sfreq),
    )[0] * 1e6
    target = first_n2 - window_end

    print("\n--- Training Example ---")
    print("Window:", window_start, "to", window_end, "seconds")
    print("EEG samples:", len(eeg_window))
    print("Target:", target, "seconds until N2")

    figure = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(eeg_window)) / sfreq, eeg_window)
    plt.xlabel("Time within window (seconds)")
    plt.ylabel("Amplitude (microvolts)")
    plt.title(f"5-second EEG window | Target: {target:.0f}s")
    plt.tight_layout()
    window_figure = save_figure(figure, figures / "eeg_5s_before_n2.png")
    print("Saved figure:", window_figure)


if __name__ == "__main__":
    main()
