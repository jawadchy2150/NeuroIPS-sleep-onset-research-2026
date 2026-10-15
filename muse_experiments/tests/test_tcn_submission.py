"""Codabench checks for the 120-second TCN."""

import copy
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from muse_experiments.io import REPO_ROOT
from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.preprocessing.windows import stack_history

SUBMISSION_PY = (
    REPO_ROOT / "muse_experiments" / "submission_tcn" / "submission.py"
)
FINAL_CHECKPOINT = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "tcn_120s_recording_loss_250train_locked17_epoch14_seed42.pt"
)
OLD_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_tcn_120s_250_recordings.zip"
)
NEW_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_tcn_120s_250_recordings_cached.zip"
)
OLD_ZIP_SHA256 = (
    "d5772374fc6b0696426533a7303c3f0595d0ac7eb1d9a4d8194f0b787c6086b6"
)


def load_submission():
    spec = importlib.util.spec_from_file_location(
        "tcn_codabench_submission", SUBMISSION_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _net():
    return SleepOnsetNet(
        n_channels=4,
        width=32,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
        uv_scale=100.0,
        use_temporal=True,
        cap_seconds=600.0,
    )


def _submission_model(module, state, meta_extra=None):
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        torch.save(state, folder / module.ARTIFACT)
        meta = {
            "submission_dir": folder,
            "sfreq": 128,
            "n_chans": 4,
            "n_times": 640,
            "device": "cpu",
            "ch_names": ["TP9", "AF7", "AF8", "TP10"],
        }
        if meta_extra:
            meta.update(meta_extra)
        return module.Solver().load_model(meta)


def _offline(model, windows):
    scores = []
    history = []
    for window in windows:
        history.append(window)
        sequence, mask = stack_history(
            [item.numpy() for item in history], 24
        )
        with torch.inference_mode():
            score = model.predict(
                torch.from_numpy(sequence).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )
        scores.append(float(score))
    return np.asarray(scores, dtype=np.float64)


class TcnSubmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = load_submission()
        torch.manual_seed(4)
        cls.state = _net().state_dict()

    def test_source_is_self_contained(self):
        text = SUBMISSION_PY.read_text(encoding="utf-8")
        self.assertNotIn("muse_experiments", text)
        self.assertNotIn("import lightgbm", text)
        self.assertNotIn('.to("mps")', text)
        self.assertNotIn(".cuda(", text)
        self.assertNotIn("1e6", text)
        self.assertIn("class Solver(CompetSolver)", text)
        self.assertEqual(self.submission.HISTORY, 24)
        self.assertEqual(self.submission.UV_SCALE, 100.0)
        self.assertEqual(
            self.submission.CHANNELS, ("TP9", "AF7", "AF8", "TP10")
        )

    def test_state_dict_matches_the_training_architecture(self):
        module = self.submission
        model = _submission_model(module, self.state, {"device": "cuda"})
        self.assertEqual(len(self.state), 28)
        if torch.cuda.is_available():
            self.assertEqual(next(model.parameters()).device.type, "cuda")
        else:
            self.assertEqual(next(model.parameters()).device.type, "cpu")
        batch = torch.randn(3, 4, 640)
        scores = model.predict(batch)
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.isfinite(scores).all())
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 600.0)
        with self.assertRaises(ValueError):
            model.predict(torch.randn(1, 4, 500))

    def test_full_recording_stream_matches_offline_history(self):
        trained = _net().eval()
        trained.load_state_dict(self.state)
        submitted = _submission_model(self.submission, self.state)
        first = torch.randn(30, 4, 640)
        second = torch.randn(8, 4, 640)
        submitted.reset_state()
        streamed = submitted.predict(first).numpy()
        offline = _offline(trained, first)
        self.assertEqual(streamed.shape, (30,))
        self.assertLess(float(np.max(np.abs(streamed - offline))), 1e-5)
        stepper = _submission_model(self.submission, self.state)
        one_by_one = np.concatenate([
            stepper.predict(first[index:index + 1]).numpy()
            for index in range(first.shape[0])
        ])
        self.assertLess(float(np.max(np.abs(one_by_one - streamed))), 1e-5)
        submitted.reset_state()
        restarted = submitted.predict(second[:1]).numpy()
        alone = _offline(trained, second[:1])
        self.assertLess(float(np.max(np.abs(restarted - alone))), 1e-5)
        leaked = _submission_model(self.submission, self.state)
        leaked.predict(first)
        without_reset = leaked.predict(second[:1]).numpy()
        self.assertGreater(float(np.max(np.abs(without_reset - alone))), 1e-4)

    def test_reset_and_deepcopy_clear_only_the_copy(self):
        model = _submission_model(self.submission, self.state)
        window = torch.randn(1, 4, 640)
        first = model.predict(window)
        model.predict(torch.randn(4, 4, 640))
        model.reset_state()
        self.assertEqual(model._embeddings, [])
        self.assertTrue(torch.allclose(first, model.predict(window)))
        model.reset_state()
        clone = copy.deepcopy(model)
        clone.predict(torch.randn(6, 4, 640))
        self.assertEqual(model._embeddings, [])
        self.assertEqual(len(clone._embeddings), 6)
        clone.reset_state()
        self.assertEqual(clone._embeddings, [])
        left = model.predict(window)
        clone_window = clone.predict(window)
        self.assertTrue(torch.allclose(left, clone_window))

    def test_channel_order_and_clip(self):
        module = self.submission
        model = _submission_model(module, self.state)
        batch = torch.randn(2, 4, 640)
        permuted = batch[:, torch.tensor([3, 2, 1, 0]), :]
        reordered = _submission_model(
            module,
            self.state,
            {"ch_names": ["TP10", "AF8", "AF7", "TP9"]},
        )
        self.assertTrue(torch.allclose(
            model.predict(batch),
            reordered.predict(permuted),
            atol=1e-5,
        ))
        model.head.weight.data.zero_()
        model.head.bias.data.fill_(10_000.0)
        model.reset_state()
        high = model.predict(torch.zeros(1, 4, 640))
        self.assertEqual(float(high), 600.0)
        model.head.bias.data.fill_(float("nan"))
        model.reset_state()
        fallback = model.predict(torch.zeros(1, 4, 640))
        self.assertEqual(float(fallback), 300.0)

    def test_mps_matches_cpu_when_available(self):
        if not torch.backends.mps.is_available():
            self.skipTest("mps is not available")
        cpu_model = _submission_model(
            self.submission, self.state, {"device": "cpu"}
        )
        mps_model = _submission_model(
            self.submission, self.state, {"device": "mps"}
        )
        self.assertEqual(next(mps_model.parameters()).device.type, "mps")
        batch = torch.randn(40, 4, 640)
        cpu_scores = cpu_model.predict(batch)
        mps_scores = mps_model.predict(batch)
        self.assertEqual(mps_scores.device.type, "cpu")
        gap = float((cpu_scores - mps_scores).abs().max())
        self.assertLess(gap, 1e-4)


class FinalRecordingTest(unittest.TestCase):
    def test_validation_recordings_match_the_training_model(self):
        if not FINAL_CHECKPOINT.is_file():
            self.skipTest("final checkpoint is not trained yet")
        import pandas as pd

        from muse_experiments.config import load_config
        from muse_experiments.evaluation.metrics import weighted_bmae
        from muse_experiments.io import repo_path
        from muse_experiments.preprocessing.windows import (
            ensure_window_cache,
            load_feature_table,
        )
        from muse_experiments.training.lightgbm_run import split_subjects

        module = load_submission()
        blob = torch.load(
            FINAL_CHECKPOINT, map_location="cpu", weights_only=False
        )
        state = blob["state_dict"]
        self.assertEqual(int(blob["epoch"]), 14)
        trained = _net().eval()
        trained.load_state_dict(state)
        submitted = _submission_model(module, state)
        cfg = load_config("cnn_5s_recording_loss_250")
        frame = load_feature_table(cfg)
        _train_idx, val_idx = split_subjects(frame, cfg)
        cache = ensure_window_cache(
            frame, cfg, repo_path(cfg["paths"]["muse_data_root"])
        )
        validation = frame.iloc[val_idx].reset_index(drop=True)
        order = np.array(val_idx)
        keys = list(zip(
            frame["subject"].astype(str).to_numpy()[order],
            frame["session"].astype(str).to_numpy()[order],
        ))
        predictions = []
        references = []
        cursor = 0
        while cursor < len(keys):
            subject, session = keys[cursor]
            stop = cursor + 1
            while stop < len(keys) and keys[stop] == (subject, session):
                stop += 1
            rows = order[cursor:stop]
            windows = []
            for row in rows:
                sample = np.array(
                    cache.array[row], dtype=np.float32, copy=True
                )
                windows.append(torch.from_numpy(sample))
            submitted.reset_state()
            streamed = submitted.predict(torch.stack(windows)).numpy()
            offline = _offline(trained, windows)
            self.assertLess(
                float(np.max(np.abs(streamed - offline))), 1e-5
            )
            predictions.append(streamed)
            references.append(offline)
            cursor = stop
        prediction = np.concatenate(predictions)
        reference = np.concatenate(references)
        self.assertEqual(prediction.shape, (2603,))
        self.assertLess(float(np.max(np.abs(prediction - reference))), 1e-5)
        score = weighted_bmae(
            validation,
            prediction,
            [0, 40, 90, 300, 600],
            [10, 5, 3, 1],
        )
        report = REPO_ROOT / (
            "muse_experiments/results/analysis/"
            "tcn_120s_250_locked17_eval.json"
        )
        if report.is_file():
            import json
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertLess(abs(score - saved["tcn"]["wbmae"]), 0.02)
        self.assertTrue(np.isfinite(prediction).all())


class ZipContentsTest(unittest.TestCase):
    def test_previous_zip_is_unchanged(self):
        digest = hashlib.sha256(OLD_ZIP.read_bytes()).hexdigest()
        self.assertEqual(digest, OLD_ZIP_SHA256)

    def test_packaged_zip_if_present(self):
        if not NEW_ZIP.is_file():
            self.skipTest("zip is created after the score check")
        with zipfile.ZipFile(NEW_ZIP) as archive:
            names = archive.namelist()
            weights = archive.read("tcn_weights.pt")
            source = archive.read("submission.py")
        self.assertEqual(sorted(names), ["submission.py", "tcn_weights.pt"])
        self.assertTrue(all("/" not in name for name in names))
        self.assertEqual(source, SUBMISSION_PY.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tcn_weights.pt"
            path.write_bytes(weights)
            state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(len(state), 28)


if __name__ == "__main__":
    unittest.main()
