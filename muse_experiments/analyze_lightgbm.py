"""Compatibility entry point for LightGBM error analysis.

The implementation is ``muse_experiments.evaluation.analyze_lightgbm``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.evaluation.analyze_lightgbm import main

if __name__ == "__main__":
    main()
