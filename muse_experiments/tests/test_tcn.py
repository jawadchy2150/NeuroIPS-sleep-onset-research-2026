"""Synthetic checks for the causal temporal CNN."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.preprocessing.windows import (
    SequenceDataset,
    WindowCache,
    history_bounds,
    recording_starts,
    stack_history,
)
from muse_experiments.tcn.streaming import RecordingBuffer


def _net(**kwargs):
    defaults = dict(
        n_channels=4,
        width=8,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.0,
        uv_scale=100.0,
        use_temporal=True,
    )
    defaults.update(kwargs)
    return SleepOnsetNet(**defaults)


class SequenceTests(unittest.TestCase):
    def test_history_stays_inside_the_recording(self):
        subjects = ["a", "a", "a", "b", "b"]
        sessions = ["s", "s", "s", "t", "t"]
        starts = recording_starts(subjects, sessions)
        self.assertEqual(history_bounds(starts[0], 0, 4), (0, 0))
        self.assertEqual(history_bounds(starts[2], 2, 4), (0, 2))
        self.assertEqual(history_bounds(starts[3], 3, 4), (3, 3))
        self.assertEqual(history_bounds(starts[4], 4, 4), (3, 4))

    def test_dataset_does_not_cross_recordings(self):
        windows = np.arange(5 * 4 * 8, dtype=np.float32).reshape(5, 4, 8)
        frame = pd.DataFrame({
            "subject": ["a", "a", "a", "b", "b"],
            "session": ["s", "s", "s", "t", "t"],
            "target_s": [10, 20, 30, 40, 50],
        })
        cache = WindowCache("memory", windows)
        dataset = SequenceDataset(cache, frame, [3], history=4)
        sequence, mask, target, subject, session = dataset[0]
        self.assertEqual(subject, "b")
        self.assertEqual(session, "t")
        self.assertEqual(float(target), 40.0)
        self.assertEqual(tuple(sequence.shape), (4, 4, 8))
        self.assertTrue(np.array_equal(mask.numpy(), [0, 0, 0, 1]))
        self.assertTrue(
            np.array_equal(sequence[-1].numpy(), windows[3])
        )
        self.assertEqual(float(sequence[:-1].abs().sum()), 0.0)

    def test_stack_history_left_pads(self):
        window = np.ones((4, 8), dtype=np.float32)
        sequence, mask = stack_history([window], 3)
        self.assertEqual(tuple(sequence.shape), (3, 4, 8))
        self.assertTrue(np.array_equal(mask, [0, 0, 1]))


class CausalityTests(unittest.TestCase):
    def test_future_windows_do_not_change_earlier_predictions(self):
        torch.manual_seed(0)
        model = _net().eval()
        sequence = torch.randn(2, 6, 4, 640)
        sequence[1, :4] = sequence[0, :4]
        sequence[1, 4:] = torch.randn(2, 4, 640)
        mask = torch.ones(2, 6)
        early = model.forward_steps(sequence[:, :4], mask[:, :4])
        self.assertTrue(torch.allclose(early[0], early[1], atol=1e-5))
        full = model.forward_steps(sequence, mask)
        self.assertTrue(torch.allclose(full[:, :4], early, atol=1e-5))

    def test_padding_mask_blocks_filler_values(self):
        torch.manual_seed(1)
        model = _net().eval()
        sequence = torch.randn(1, 4, 4, 640)
        mask = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        other = sequence.clone()
        other[:, :3] = 50.0
        first = model(sequence, mask)
        second = model(other, mask)
        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_output_shape_and_clip(self):
        model = _net().eval()
        sequence = torch.randn(2, 5, 4, 640)
        mask = torch.ones(2, 5)
        raw = model(sequence, mask)
        clipped = model.predict(sequence, mask)
        self.assertEqual(tuple(raw.shape), (2,))
        self.assertEqual(tuple(clipped.shape), (2,))
        self.assertTrue(torch.isfinite(clipped).all())
        self.assertTrue(torch.all(clipped >= 0))
        self.assertTrue(torch.all(clipped <= 600))


class StreamTests(unittest.TestCase):
    def test_stream_matches_the_batched_prediction(self):
        torch.manual_seed(2)
        model = _net().eval()
        windows = torch.randn(5, 4, 640)
        buffer = RecordingBuffer(model, history=4)
        streamed = buffer.predict(windows)
        sequence, mask = stack_history(
            [windows[index].numpy() for index in range(5)], 4
        )
        batched = model.predict(
            torch.from_numpy(sequence).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )
        self.assertEqual(tuple(streamed.shape), (5,))
        self.assertTrue(np.isfinite(streamed).all())
        self.assertAlmostEqual(
            float(streamed[-1]), float(batched[0]), places=4
        )

    def test_reset_and_deepcopy_drop_only_the_copy(self):
        torch.manual_seed(3)
        model = _net().eval()
        buffer = RecordingBuffer(model, history=3)
        windows = torch.randn(2, 4, 640)
        first = buffer.predict(windows[:1])
        self.assertEqual(len(buffer._windows), 1)
        clone = copy.deepcopy(buffer)
        clone.reset_state()
        self.assertEqual(len(buffer._windows), 1)
        self.assertEqual(len(clone._windows), 0)
        replay = clone.predict(windows[:1])
        self.assertAlmostEqual(float(first[0]), float(replay[0]), places=4)

    def test_cpu_and_available_accelerator(self):
        model = _net().eval()
        sequence = torch.randn(1, 3, 4, 640)
        mask = torch.ones(1, 3)
        cpu = model.predict(sequence, mask)
        self.assertEqual(cpu.device.type, "cpu")
        if torch.backends.mps.is_available():
            moved = _net().to("mps").eval()
            out = moved.predict(sequence.to("mps"), mask.to("mps"))
            self.assertEqual(out.device.type, "mps")
            self.assertTrue(torch.isfinite(out).all())


class TrainInferTests(unittest.TestCase):
    def test_eval_prediction_matches_a_second_call(self):
        torch.manual_seed(4)
        model = _net()
        sequence = torch.randn(2, 4, 4, 640)
        mask = torch.ones(2, 4)
        model.eval()
        first = model.predict(sequence, mask)
        second = model.predict(sequence, mask)
        self.assertTrue(torch.allclose(first, second))

    def test_synthetic_fit_writes_only_the_requested_file(self):
        from muse_experiments.training.tcn_run import _run_epoch

        torch.manual_seed(5)
        model = _net()
        sequence = torch.randn(4, 3, 4, 640)
        mask = torch.ones(4, 3)
        target = torch.tensor([10.0, 50.0, 120.0, 400.0])
        loader = [(
            sequence,
            mask,
            target,
            ["s", "s", "t", "t"],
            ["a", "a", "b", "b"],
        )]
        cfg = {
            "evaluation": {
                "bins": [0, 40, 90, 300, 600],
                "bin_weights": [10, 5, 3, 1],
                "cap_seconds": 600,
            },
            "training": {"grad_clip": 1.0},
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = [value.detach().clone() for value in model.parameters()]
        _run_epoch(model, loader, torch.device("cpu"), optimizer, cfg, 1.0)
        changed = any(
            not torch.allclose(old, new)
            for old, new in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        model.eval()
        prediction = model.predict(sequence, mask)
        self.assertEqual(tuple(prediction.shape), (4,))
        self.assertTrue(torch.isfinite(prediction).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
