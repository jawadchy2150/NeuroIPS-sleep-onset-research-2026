"""Codabench checks for the 250-recording 5-second CNN."""

import copy
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from muse_experiments.config import load_config
from muse_experiments.evaluation.metrics import weighted_bmae
from muse_experiments.io import REPO_ROOT, repo_path
from muse_experiments.models.temporal import SleepOnsetNet
from muse_experiments.preprocessing.windows import (
    ensure_window_cache,
    load_feature_table,
)
from muse_experiments.training.lightgbm_run import split_subjects

EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]
RECORDED_WBMAE = 50.7552948263006
CHECKPOINT = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "cnn_5s_recording_loss_250train_locked17_seed42.pt"
)
PREVIOUS_CHECKPOINT = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "cnn_5s_recording_loss_100_pre_n2_split33_seed42.pt"
)
SUBMISSION_PY = (
    REPO_ROOT / "muse_experiments" / "submission_cnn" / "submission.py"
)
PREVIOUS_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_cnn_recording_loss.zip"
)
LIGHTGBM_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_weighted_lightgbm_pre_n2.zip"
)
NEW_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_cnn_250_recordings.zip"
)


def load_submission():
    spec = importlib.util.spec_from_file_location(
        "cnn_250_codabench_submission", SUBMISSION_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict():
    blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return blob["state_dict"]


def _training_model(state):
    model = SleepOnsetNet(
        n_channels=4,
        width=32,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.1,
        uv_scale=100.0,
        use_temporal=False,
        cap_seconds=600.0,
    )
    model.load_state_dict(state)
    model.to("cpu")
    model.eval()
    return model


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


def _validation_predictions(module, state):
    cfg = load_config("cnn_5s_recording_loss_250")
    frame = load_feature_table(cfg)
    _train_idx, val_idx = split_subjects(frame, cfg)
    cache = ensure_window_cache(
        frame,
        cfg,
        repo_path(cfg["paths"]["muse_data_root"]),
    )
    submitted = _submission_model(module, state)
    trained = _training_model(state)
    validation = frame.iloc[val_idx].reset_index(drop=True)
    predictions = []
    references = []
    for start in range(0, len(val_idx), 64):
        rows = val_idx[start:start + 64]
        windows = torch.as_tensor(
            np.asarray(cache.array[rows]), dtype=torch.float32
        )
        predictions.append(submitted.predict(windows).numpy())
        with torch.inference_mode():
            references.append(
                trained.predict(
                    windows.unsqueeze(1),
                    torch.ones(windows.shape[0], 1),
                ).numpy()
            )
    return validation, np.concatenate(predictions), np.concatenate(references)


class Cnn250SubmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = load_submission()
        cls.state = _state_dict()
        cls.before = {
            "checkpoint": _sha256(CHECKPOINT),
            "previous_checkpoint": _sha256(PREVIOUS_CHECKPOINT),
            "submission": _sha256(SUBMISSION_PY),
            "previous_zip": _sha256(PREVIOUS_ZIP),
            "lightgbm_zip": _sha256(LIGHTGBM_ZIP),
        }

    def tearDown(self):
        self.assertEqual(_sha256(CHECKPOINT), self.before["checkpoint"])
        self.assertEqual(
            _sha256(PREVIOUS_CHECKPOINT),
            self.before["previous_checkpoint"],
        )
        self.assertEqual(_sha256(SUBMISSION_PY), self.before["submission"])
        self.assertEqual(_sha256(PREVIOUS_ZIP), self.before["previous_zip"])
        self.assertEqual(
            _sha256(LIGHTGBM_ZIP), self.before["lightgbm_zip"]
        )

    def test_source_is_the_previous_cpu_submission(self):
        text = SUBMISSION_PY.read_text(encoding="utf-8")
        self.assertNotIn("muse_experiments", text)
        self.assertNotIn("import lightgbm", text)
        self.assertNotIn('.to("mps")', text)
        self.assertNotIn(".cuda(", text)
        self.assertNotIn("1e6", text)
        self.assertIn("class Solver(CompetSolver)", text)
        self.assertEqual(
            self.submission.CHANNELS, ("TP9", "AF7", "AF8", "TP10")
        )
        self.assertEqual(self.submission.SAMPLES, 640)
        self.assertEqual(self.submission.UV_SCALE, 100.0)
        self.assertEqual(self.submission.CAP_S, 600.0)
        self.assertEqual(len(self.state), 12)

    def test_contract_shapes_clip_and_channel_order(self):
        module = self.submission
        model = _submission_model(module, self.state, {"device": "cuda"})
        batch = torch.randn(3, 4, 640)
        scores = model.predict(batch)
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.isfinite(scores).all())
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 600.0)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

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
        with self.assertRaises(ValueError):
            model.predict(torch.randn(2, 4, 500))
        with self.assertRaises(FileNotFoundError):
            module.Solver().load_model({
                "submission_dir": Path(tempfile.mkdtemp()),
                "sfreq": 128,
            })
        with self.assertRaises(ValueError):
            _submission_model(module, self.state, {"sfreq": 100})

        model.head.weight.data.zero_()
        model.head.bias.data.fill_(10_000.0)
        high = model.predict(torch.zeros(2, 4, 640))
        self.assertTrue(torch.equal(high, torch.full((2,), 600.0)))
        model.head.bias.data.fill_(-10_000.0)
        low = model.predict(torch.zeros(2, 4, 640))
        self.assertTrue(torch.equal(low, torch.zeros(2)))
        model.head.bias.data.fill_(float("nan"))
        fallback = model.predict(torch.zeros(1, 4, 640))
        self.assertEqual(float(fallback), 300.0)

    def test_streaming_reset_and_deepcopy(self):
        model = _submission_model(self.submission, self.state)
        windows = torch.randn(5, 4, 640)
        batched = model.predict(windows)
        streamed = torch.cat([
            model.predict(windows[index:index + 1])
            for index in range(windows.shape[0])
        ])
        self.assertTrue(torch.allclose(batched, streamed, atol=0.0))
        first = model.predict(windows[:1])
        model.reset_state()
        self.assertTrue(torch.allclose(
            first, model.predict(windows[:1]), atol=0.0
        ))
        worker = copy.deepcopy(model)
        worker.reset_state()
        self.assertTrue(torch.allclose(
            model.predict(windows[:1]),
            worker.predict(windows[:1]),
            atol=0.0,
        ))

    def test_matches_saved_model_and_validation_score(self):
        validation, prediction, reference = _validation_predictions(
            self.submission, self.state
        )
        self.assertEqual(prediction.shape, (2603,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 600.0)
        self.assertLess(float(np.max(np.abs(prediction - reference))), 1e-5)
        score = weighted_bmae(validation, prediction, EDGES, WEIGHTS)
        self.assertLess(abs(score - RECORDED_WBMAE), 0.02)


class ZipContentsTest(unittest.TestCase):
    def test_packaged_zip_if_present(self):
        if not NEW_ZIP.is_file():
            self.skipTest("zip is created after the score check")
        with zipfile.ZipFile(NEW_ZIP) as archive:
            names = archive.namelist()
            weights = archive.read("cnn_weights.pt")
            source = archive.read("submission.py")
        self.assertEqual(sorted(names), ["cnn_weights.pt", "submission.py"])
        self.assertTrue(all("/" not in name for name in names))
        self.assertEqual(source, SUBMISSION_PY.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cnn_weights.pt"
            path.write_bytes(weights)
            state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(set(state), set(_state_dict()))
        for key, value in _state_dict().items():
            self.assertTrue(torch.equal(state[key], value))


if __name__ == "__main__":
    unittest.main()
