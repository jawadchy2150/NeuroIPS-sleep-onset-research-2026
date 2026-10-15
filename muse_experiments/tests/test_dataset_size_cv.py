"""Subject folds stay disjoint and the 100-recording set is a subset."""

import unittest

from muse_experiments.splits import (
    assign_subject_folds,
    round_robin_recordings,
    select_training_recordings,
)


class DatasetSizeCvTest(unittest.TestCase):
    def test_folds_cover_each_subject_once(self):
        subjects = [f"sub-{index:03d}" for index in range(1, 21)]
        first = assign_subject_folds(subjects, n_splits=5, seed=42)
        second = assign_subject_folds(list(reversed(subjects)), 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(subjects))
        counts = {fold: 0 for fold in range(1, 6)}
        for fold in first.values():
            counts[fold] += 1
        self.assertEqual(sum(counts.values()), 20)
        self.assertTrue(all(count == 4 for count in counts.values()))

    def test_round_robin_orders_sessions_numerically(self):
        pairs = [
            ("sub-002", "ses-010"),
            ("sub-002", "ses-002"),
            ("sub-001", "ses-002"),
            ("sub-001", "ses-001"),
        ]
        ordered = round_robin_recordings(pairs)
        self.assertEqual(ordered, [
            ("sub-001", "ses-001"),
            ("sub-002", "ses-002"),
            ("sub-001", "ses-002"),
            ("sub-002", "ses-010"),
        ])

    def test_smaller_set_is_a_prefix_and_skips_blocked_subjects(self):
        pairs = [
            ("sub-001", "ses-001"),
            ("sub-001", "ses-002"),
            ("sub-004", "ses-001"),
            ("sub-003", "ses-001"),
            ("sub-003", "ses-002"),
        ]
        blocked = ["sub-004"]
        large = select_training_recordings(pairs, blocked, 3)
        small = select_training_recordings(pairs, blocked, 2)
        self.assertEqual(small, large[:2])
        self.assertNotIn("sub-004", {subject for subject, _session in large})
        with self.assertRaises(ValueError):
            select_training_recordings(pairs, blocked, 5)


if __name__ == "__main__":
    unittest.main()
