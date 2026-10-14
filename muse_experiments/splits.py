"""Participant-level splits for team experiments.

Windows from one subject stay in a single split. The Muse warm-up cut is
``official_muse_split``: subjects who have an official test recording stay
in training, and 20% of the remaining subjects are held out with
``train_test_split`` at ``random_state=33``. ``participant_split`` remains
for older GroupShuffleSplit checks.

The official Muse ``split`` column is the competition boundary. Do not
train or tune on recordings marked test.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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


def official_muse_split(
    subjects, has_official_test, val_size=0.2, seed=33
):
    """Validation cut used by the Muse warm-up config.

    Subjects who appear in the official test split stay in training.
    Of the remaining subjects, sorted for stability, ``val_size`` go to
    validation via ``train_test_split(..., random_state=seed)``. That is
    the same call as NeuralBench ``PredefinedSplit`` with
    ``valid_random_state``.
    """
    subjects = np.asarray(subjects)
    has_test = np.asarray(has_official_test, dtype=bool)
    if has_test.shape != subjects.shape:
        raise ValueError("has_official_test must align with subjects")
    eligible = sorted(set(subjects[~has_test].tolist()))
    if len(eligible) < 2:
        raise ValueError(
            "official validation needs at least two subjects "
            "without a test recording"
        )
    _train_subjects, valid_subjects = train_test_split(
        np.asarray(eligible),
        test_size=val_size,
        random_state=seed,
    )
    valid_ids = set(valid_subjects.tolist())
    val_mask = np.array(
        [
            (subject in valid_ids) and (not flagged)
            for subject, flagged in zip(subjects, has_test)
        ]
    )
    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask)
    assert_disjoint(subjects[train_idx], subjects[val_idx])
    return train_idx, val_idx
