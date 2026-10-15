"""The 120-second TCN can use the recording-level W-bMAE loss."""

import unittest

import numpy as np
import pandas as pd
import torch

from muse_experiments.config import load_config
from muse_experiments.io import REPO_ROOT
from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.splits import assign_subject_folds
from muse_experiments.training.recording_loss import (
    chunk_loss,
    recording_wbmae_loss,
)

EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]


class _History:
    def __init__(self, targets):
        self.targets = list(targets)

    def __getitem__(self, index):
        sequence = torch.zeros(4, 4, 640)
        sequence[-1] = float(index)
        mask = torch.tensor([0.0, 0.0, 0.0, 1.0])
        target = torch.tensor(float(self.targets[index]))
        return sequence, mask, target, "sub", "ses"


def _temporal():
    return SleepOnsetNet(
        n_channels=4,
        width=8,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.0,
        uv_scale=100.0,
        use_temporal=True,
        cap_seconds=600.0,
    )


class TemporalLossTest(unittest.TestCase):
    def test_chunk_loss_averages_recordings(self):
        torch.manual_seed(0)
        model = _temporal().eval()
        dataset = _History([10.0, 20.0, 100.0, 400.0, 500.0])
        groups = [[0, 1], [2, 3, 4]]
        total = chunk_loss(
            model, dataset, groups, torch.device("cpu"), EDGES, WEIGHTS
        )
        singles = [
            chunk_loss(
                model, dataset, [group], torch.device("cpu"), EDGES, WEIGHTS
            )
            for group in groups
        ]
        mean = torch.stack(singles).mean()
        self.assertTrue(torch.allclose(total, mean, atol=1e-5))

    def test_longer_recording_does_not_dominate_the_loss(self):
        short = recording_wbmae_loss(
            torch.full((2,), 18.0),
            torch.full((2,), 10.0),
            EDGES,
            WEIGHTS,
        )
        long = recording_wbmae_loss(
            torch.full((40,), 18.0),
            torch.full((40,), 10.0),
            EDGES,
            WEIGHTS,
        )
        self.assertAlmostEqual(float(short), float(long), places=5)

    def test_one_temporal_update_is_finite(self):
        torch.manual_seed(1)
        model = _temporal()
        dataset = _History([12.0, 30.0, 80.0, 450.0])
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
        self.assertTrue(np.isfinite(float(loss.detach())))
        model.eval()
        sequence = torch.zeros(2, 4, 4, 640)
        mask = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        first = model.predict(sequence, mask)
        second = model.predict(sequence, mask)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(tuple(first.shape), (2,))

    def test_future_context_does_not_change_an_earlier_score(self):
        torch.manual_seed(2)
        model = _temporal().eval()
        sequence = torch.randn(1, 6, 4, 640)
        mask = torch.ones(1, 6)
        early = model.forward_steps(sequence[:, :4], mask[:, :4])
        full = model.forward_steps(sequence, mask)
        self.assertTrue(torch.allclose(full[:, :4], early, atol=1e-5))

    def test_saved_folds_match_the_seed_and_tcn_settings(self):
        path = (
            REPO_ROOT
            / "muse_experiments"
            / "results"
            / "cv_dataset_size"
            / "assignments.csv"
        )
        saved = pd.read_csv(path)
        rebuilt = assign_subject_folds(
            saved["subject"].astype(str), n_splits=5, seed=42
        )
        found = {
            str(subject): int(fold)
            for subject, fold in zip(saved["subject"], saved["fold"])
        }
        self.assertEqual(found, rebuilt)
        cfg = load_config("tcn_120s")
        self.assertEqual(cfg["model"]["history_windows"], 24)
        self.assertTrue(cfg["model"]["use_temporal"])
        self.assertEqual(cfg["training"]["max_epochs"], 15)
        self.assertEqual(cfg["training"]["learning_rate"], 0.001)
        self.assertEqual(
            cfg["data"]["channels"], ["TP9", "AF7", "AF8", "TP10"]
        )
        self.assertEqual(cfg["data"]["window_samples"], 640)


if __name__ == "__main__":
    unittest.main()
