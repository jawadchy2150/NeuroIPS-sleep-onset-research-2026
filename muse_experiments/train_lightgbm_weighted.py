"""Compatibility entry point for weighted LightGBM training.

The implementation is ``muse_experiments.training.train_lightgbm_weighted``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.training.train_lightgbm_weighted import main

if __name__ == "__main__":
    main()
