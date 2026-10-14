"""Compatibility entry point for the Muse preview.

The implementation is ``muse_experiments.preprocessing.explore_muse``.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from muse_experiments.preprocessing.explore_muse import main

if __name__ == "__main__":
    main()
