"""Train the corrected 5-second CNN on 250 locked-holdout recordings."""

import platform
import resource
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.preprocessing.expand_training_recordings import (
    prepare_expanded_table,
)
from muse_experiments.training.tcn_run import main as run_main


def main():
    summary = prepare_expanded_table()
    print(
        "Training recordings:", summary["train_recordings"],
        "subjects:", summary["train_subjects"],
        "windows:", summary["train_windows"],
    )
    run_main("cnn_5s_recording_loss_250")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        rss_bytes = rss
    else:
        rss_bytes = rss * 1024
    print(f"Peak RSS: {rss_bytes / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
