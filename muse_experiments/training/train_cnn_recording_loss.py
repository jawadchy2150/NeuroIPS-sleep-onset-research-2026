"""Train the 5-second CNN with the recording-level W-bMAE loss."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.training.tcn_run import main as run_main


def main():
    run_main("cnn_5s_recording_loss")


if __name__ == "__main__":
    main()
