"""The 250-recording split keeps the original 17 subjects locked."""

import unittest

import pandas as pd

from muse_experiments.config import load_config
from muse_experiments.io import repo_path
from muse_experiments.preprocessing.expand_training_recordings import (
    choose_training_keys,
)
from muse_experiments.splits import locked_validation_split
from muse_experiments.training.lightgbm_run import split_subjects


class ExpandedSplitTest(unittest.TestCase):
    def test_locked_subjects_stay_on_one_side(self):
        subjects = np_subjects()
        train_idx, val_idx = locked_validation_split(subjects, ["b"])
        self.assertEqual(
            {subjects[index] for index in val_idx}, {"b"}
        )
        self.assertEqual(
            {subjects[index] for index in train_idx}, {"a", "c"}
        )

    def test_extra_session_of_a_holdout_subject_stays_out_of_training(self):
        subjects = ["hold", "hold", "train"]
        train_idx, val_idx = locked_validation_split(subjects, ["hold"])
        self.assertEqual(list(train_idx), [2])
        self.assertEqual(list(val_idx), [0, 1])

    def test_selection_keeps_locked_recordings_and_skips_holdout(self):
        locked = [("sub-001", "ses-001"), ("sub-002", "ses-001")]
        candidates = [
            ("sub-001", "ses-002"),
            ("sub-009", "ses-001"),
            ("sub-003", "ses-001"),
            ("sub-003", "ses-002"),
            ("sub-004", "ses-001"),
        ]
        chosen = choose_training_keys(
            candidates, locked, blocked=["sub-009"], target=4
        )
        self.assertEqual(chosen[:2], locked)
        self.assertNotIn(("sub-009", "ses-001"), chosen)
        self.assertEqual(len(chosen), 4)

    def test_config_lists_the_original_validation_subjects(self):
        old = load_config("lightgbm_weighted")
        frame = pd.read_csv(repo_path(old["inputs"]["features_csv"]))
        _train_idx, val_idx = split_subjects(frame, old)
        original = sorted(frame.iloc[val_idx]["subject"].unique())
        cfg = load_config("cnn_5s_recording_loss_250")
        listed = sorted(cfg["split"]["validation_subjects"])
        self.assertEqual(listed, original)
        self.assertEqual(cfg["training"]["loss"], "recording_wbmae")
        self.assertEqual(cfg["training"]["recordings_per_update"], 1)
        self.assertEqual(cfg["training"]["learning_rate"], 0.001)
        self.assertFalse(cfg["model"]["use_temporal"])
        self.assertEqual(cfg["model"]["history_windows"], 1)


def np_subjects():
    return ["a", "b", "b", "c", "a"]


if __name__ == "__main__":
    unittest.main()
