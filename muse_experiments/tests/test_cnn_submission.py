"""Codabench packaging checks for the 5-second CNN."""

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
RECORDED_WBMAE = 57.04140873818201
CHECKPOINT = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "models"
    / "cnn_5s_recording_loss_100_pre_n2_split33_seed42.pt"
)
LIGHTGBM_ZIP = (
    REPO_ROOT
    / "muse_experiments"
    / "results"
    / "submissions"
    / "muse_weighted_lightgbm_pre_n2.zip"
)
LIGHTGBM_SUBMISSION = (
    REPO_ROOT / "muse_experiments" / "submission" / "submission.py"
)


def load_submission():
    path = (
        REPO_ROOT
        / "muse_experiments"
        / "submission_cnn"
        / "submission.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cnn_codabench_submission", path
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


class CnnSubmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = load_submission()
        cls.state = _state_dict()
        cls.before = {
            "checkpoint": _sha256(CHECKPOINT),
            "lightgbm_submission": _sha256(LIGHTGBM_SUBMISSION),
        }
        if LIGHTGBM_ZIP.is_file():
            cls.before["lightgbm_zip"] = _sha256(LIGHTGBM_ZIP)

    def tearDown(self):
        self.assertEqual(_sha256(CHECKPOINT), self.before["checkpoint"])
        self.assertEqual(
            _sha256(LIGHTGBM_SUBMISSION),
            self.before["lightgbm_submission"],
        )
        if "lightgbm_zip" in self.before:
            self.assertEqual(
                _sha256(LIGHTGBM_ZIP), self.before["lightgbm_zip"]
            )

    def test_source_is_self_contained(self):
        text = (
            REPO_ROOT
            / "muse_experiments"
            / "submission_cnn"
            / "submission.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("muse_experiments", text)
        self.assertNotIn("import lightgbm", text)
        self.assertNotIn('.to("mps")', text)
        self.assertNotIn(".cuda(", text)
        self.assertIn("class Solver(CompetSolver)", text)

    def test_contract_shapes_clip_and_channel_order(self):
        module = self.submission
        model = _submission_model(module, self.state)
        batch = torch.randn(3, 4, 640)
        scores = model.predict(batch)
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertEqual(scores.device.type, "cpu")
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(float(scores.min()) >= 0.0)
        self.assertTrue(float(scores.max()) <= 600.0)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

        flipped = torch.tensor([3, 2, 1, 0])
        permuted = batch[:, flipped, :]
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

    def test_reset_does_not_change_predictions(self):
        model = _submission_model(self.submission, self.state)
        window = torch.randn(1, 4, 640)
        first = model.predict(window)
        other = model.predict(torch.randn(4, 4, 640))
        self.assertEqual(tuple(other.shape), (4,))
        model.reset_state()
        again = model.predict(window)
        self.assertTrue(torch.allclose(first, again, atol=0.0))
        clone = copy.deepcopy(model)
        clone.reset_state()
        self.assertTrue(torch.allclose(
            model.predict(window), clone.predict(window), atol=0.0
        ))

    def test_matches_training_model_on_cpu(self):
        module = self.submission
        trained = _training_model(self.state)
        submitted = _submission_model(module, self.state)
        windows = torch.randn(16, 4, 640)
        with torch.inference_mode():
            reference = trained.predict(
                windows.unsqueeze(1),
                torch.ones(16, 1),
            )
        estimate = submitted.predict(windows)
        self.assertTrue(torch.allclose(reference, estimate, atol=1e-6))

    def test_validation_wbmae_matches_the_recorded_score(self):
        cfg = load_config("cnn_5s_recording_loss")
        frame = load_feature_table(cfg)
        _train_idx, val_idx = split_subjects(frame, cfg)
        cache = ensure_window_cache(
            frame,
            cfg,
            repo_path(cfg["paths"]["muse_data_root"]),
        )
        submitted = _submission_model(self.submission, self.state)
        trained = _training_model(self.state)
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
        prediction = np.concatenate(predictions)
        reference = np.concatenate(references)
        self.assertEqual(prediction.shape, (2603,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(float(np.max(np.abs(prediction - reference))), 1e-5)
        score = weighted_bmae(validation, prediction, EDGES, WEIGHTS)
        self.assertLess(abs(score - RECORDED_WBMAE), 0.02)


class ZipContentsTest(unittest.TestCase):
    def test_packaged_zip_if_present(self):
        path = (
            REPO_ROOT
            / "muse_experiments"
            / "results"
            / "submissions"
            / "muse_cnn_recording_loss.zip"
        )
        if not path.is_file():
            self.skipTest("zip is created after the score check")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        self.assertEqual(
            sorted(names), ["cnn_weights.pt", "submission.py"]
        )
        self.assertTrue(all("/" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
