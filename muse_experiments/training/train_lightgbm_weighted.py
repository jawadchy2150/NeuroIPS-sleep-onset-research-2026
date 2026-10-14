"""Train the weighted LightGBM config."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.training.lightgbm_run import main as run_main


def main():
    run_main("lightgbm_weighted")


if __name__ == "__main__":
    main()
