"""Offset calibration uses only the recordings it is given."""

import unittest

import numpy as np
import pandas as pd

from muse_experiments.evaluation.calibration import (
    apply_offset,
    offset_grid,
    select_offset,
)

EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]


def _frame(subject, targets, predictions):
    return pd.DataFrame({
        "subject": subject,
        "session": "ses-001",
        "target_s": targets,
        "prediction": predictions,
    })


class OffsetCalibrationTest(unittest.TestCase):
    def test_grid_matches_the_reported_search(self):
        grid = offset_grid()
        expected = np.linspace(-120.0, 40.0, 161)
        self.assertTrue(np.allclose(grid, expected))
        self.assertIn(-16.0, grid.tolist())

    def test_offset_is_clipped_after_the_shift(self):
        shifted = apply_offset([5.0, 590.0, 100.0], -16.0)
        self.assertEqual(shifted.tolist(), [0.0, 574.0, 84.0])
        high = apply_offset([590.0], 20.0)
        self.assertEqual(high.tolist(), [600.0])

    def test_selection_uses_only_the_given_frame(self):
        near = _frame(
            ["sub-a"] * 4,
            [10.0, 10.0, 10.0, 10.0],
            [26.0, 26.0, 26.0, 26.0],
        )
        later = _frame(
            ["sub-b"] * 4,
            [110.0, 110.0, 110.0, 110.0],
            [100.0, 100.0, 100.0, 100.0],
        )
        offset, score, _curve = select_offset(
            near,
            near["prediction"],
            EDGES,
            WEIGHTS,
            offset_grid(),
        )
        self.assertEqual(offset, -16.0)
        self.assertAlmostEqual(score, 0.0)
        other_offset, _other_score, _other_curve = select_offset(
            later,
            later["prediction"],
            EDGES,
            WEIGHTS,
            offset_grid(),
        )
        self.assertEqual(other_offset, 10.0)

    def test_first_grid_point_wins_a_tie(self):
        frame = _frame(["sub-a"] * 2, [0.0, 0.0], [0.0, 0.0])
        offset, _score, _curve = select_offset(
            frame,
            frame["prediction"],
            EDGES,
            WEIGHTS,
            [-20.0, -10.0, 0.0],
        )
        self.assertEqual(offset, -20.0)


if __name__ == "__main__":
    unittest.main()
