"""Recording loss matches W-bMAE and gives each recording equal weight."""

import unittest

import numpy as np
import pandas as pd
import torch

from muse_experiments.evaluation.metrics import binned_mae, weighted_bmae
from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.training.recording_loss import (
    chunk_loss,
    chunk_recordings,
    recording_groups,
    recording_wbmae_loss,
)

EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]


class _Groups:
    def __init__(self, indices, subjects, sessions):
        self.indices = np.asarray(indices)
        self.subjects = np.asarray(subjects)
        self.sessions = np.asarray(sessions)


class _Windows:
    def __init__(self, targets):
        self.targets = list(targets)

    def __getitem__(self, index):
        sequence = torch.zeros(1, 4, 640)
        mask = torch.ones(1)
        target = torch.tensor(float(self.targets[index]))
        return sequence, mask, target, "sub", "ses"


class RecordingLossTest(unittest.TestCase):
    def test_loss_matches_official_binned_mae(self):
        target = torch.tensor([0.0, 40.0, 90.0, 300.0, 600.0])
        prediction = torch.tensor([18.0, 50.0, 90.0, 280.0, 600.0])
        loss = recording_wbmae_loss(prediction, target, EDGES, WEIGHTS)
        official = binned_mae(
            target.numpy(), prediction.detach().numpy(), EDGES, WEIGHTS
        )
        self.assertAlmostEqual(float(loss), official, places=5)

    def test_missing_bin_is_left_out_of_the_denominator(self):
        target = torch.tensor([0.0, 10.0, 50.0, 100.0])
        prediction = torch.tensor([18.0, 10.0, 50.0, 100.0])
        loss = recording_wbmae_loss(prediction, target, EDGES, WEIGHTS)
        # Bin weights 10, 5, and 3 are present. The 300–600 bin is not.
        # MAE is 9, 0, and 0, so the score is 90 / 18.
        self.assertAlmostEqual(float(loss), 5.0, places=5)
        included = binned_mae(
            target.numpy(), prediction.numpy(), EDGES, WEIGHTS
        )
        self.assertAlmostEqual(float(loss), included, places=5)
        self.assertNotAlmostEqual(float(loss), 90.0 / 19.0, places=5)

    def test_mean_over_recordings_matches_wbmae(self):
        first_target = torch.tensor([10.0, 10.0])
        first_pred = torch.tensor([12.0, 16.0])
        second_target = torch.tensor([100.0, 120.0, 140.0, 160.0])
        second_pred = torch.tensor([80.0, 120.0, 150.0, 200.0])
        first = recording_wbmae_loss(
            first_pred, first_target, EDGES, WEIGHTS
        )
        second = recording_wbmae_loss(
            second_pred, second_target, EDGES, WEIGHTS
        )
        mean = float((first + second) / 2)
        frame = pd.DataFrame({
            "subject": ["a", "a", "b", "b", "b", "b"],
            "session": ["s"] * 6,
            "target_s": torch.cat([first_target, second_target]).numpy(),
        })
        prediction = torch.cat([first_pred, second_pred]).numpy()
        official = weighted_bmae(frame, prediction, EDGES, WEIGHTS)
        self.assertAlmostEqual(mean, official, places=5)

    def test_window_count_does_not_change_the_gradient(self):
        bias = torch.tensor(1.0, requires_grad=True)
        short = recording_wbmae_loss(
            bias + torch.zeros(2),
            torch.zeros(2),
            EDGES,
            WEIGHTS,
        )
        long = recording_wbmae_loss(
            bias + torch.zeros(30),
            torch.full((30,), 10.0),
            EDGES,
            WEIGHTS,
        )
        ((short + long) / 2).backward()
        self.assertAlmostEqual(float(bias.grad), 0.0, places=5)

    def test_groups_follow_recording_boundaries(self):
        dataset = _Groups(
            [0, 1, 2, 4],
            ["a", "a", "b", "b", "b"],
            ["s", "s", "s", "s", "t"],
        )
        groups = recording_groups(dataset)
        self.assertEqual(groups, [[0, 1], [2], [3]])

    def test_chunks_cover_every_recording_once(self):
        groups = [[0], [1], [2], [3], [4]]
        seen = []
        for chunk in chunk_recordings(groups, [4, 0, 1, 2, 3], 2):
            self.assertLessEqual(len(chunk), 2)
            seen.extend(chunk)
        self.assertEqual(seen, [[4], [0], [1], [2], [3]])

    def test_update_is_finite_and_repeatable(self):
        def once(seed):
            torch.manual_seed(seed)
            model = SleepOnsetNet(
                n_channels=4,
                width=32,
                use_temporal=False,
                cap_seconds=600,
            )
            dataset = _Windows([10.0, 20.0, 400.0, 500.0])
            before = [
                parameter.detach().clone()
                for parameter in model.parameters()
            ]
            loss = chunk_loss(
                model,
                dataset,
                [[0, 1], [2, 3]],
                torch.device("cpu"),
                EDGES,
                WEIGHTS,
            )
            loss.backward()
            torch.optim.AdamW(model.parameters(), lr=0.01).step()
            moved = [
                not torch.equal(old, parameter)
                for old, parameter in zip(before, model.parameters())
            ]
            return float(loss.detach()), any(moved)

        first, moved = once(7)
        second, _moved_again = once(7)
        self.assertTrue(np.isfinite(first))
        self.assertTrue(moved)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
