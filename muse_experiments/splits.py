"""Participant-level splits for team experiments.

Windows from one subject stay in a single split. This matches the
``GroupShuffleSplit`` already used by the LightGBM scripts
(``test_size=0.20``, ``random_state=42``, groups = subject).

The official Muse ``split`` column and the Sleep-EDF subject splits are
the competition boundaries. Use those for anything that will be compared
with the benchmark. This helper is only for a validation cut inside data
the official split already marked as train.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def assert_disjoint(*groups):
    """Raise if any subject id appears in more than one split."""
    seen = []
    for index, raw in enumerate(groups):
        ids = set(np.asarray(raw).tolist())
        for other_index, other in enumerate(seen):
            overlap = ids & other
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    "subject overlap between splits "
                    f"{other_index} and {index}: {sample}"
                )
        seen.append(ids)


def participant_split(groups, val_size=0.2, seed=42):
    """Return ``(train_idx, val_idx)`` with no shared subject.

    ``groups`` is one subject id per window, in the same row order as the
    feature table.
    """
    groups = np.asarray(groups)
    if groups.size == 0:
        raise ValueError("groups is empty")
    if len(np.unique(groups)) < 2:
        raise ValueError("participant split needs at least two subjects")
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(groups, groups=groups))
    assert_disjoint(groups[train_idx], groups[val_idx])
    return train_idx, val_idx
