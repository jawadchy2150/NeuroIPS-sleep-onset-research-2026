"""Synthetic checks for the corrected Muse warm-up contract."""

import copy
import importlib.util
import inspect
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from muse_experiments.config import load_config
from muse_experiments.evaluation.metrics import binned_mae, weighted_bmae
from muse_experiments.io import REPO_ROOT
from muse_experiments.models.booster import artifact_from_payload
from muse_experiments.preprocessing.features import (
    extract_features as training_features,
)
from muse_experiments.preprocessing.features import (
    pre_n2_window_bounds,
    remaining_seconds,
)
from muse_experiments.preprocessing.prepare_muse_features import (
    get_split,
    subject_has_official_test,
)
from muse_experiments.splits import assert_disjoint, official_muse_split

EDGES = [0, 40, 90, 300, 600]
WEIGHTS = [10, 5, 3, 1]


def load_submission():
    path = (
        REPO_ROOT / "muse_experiments" / "submission" / "submission.py"
    )
    spec = importlib.util.spec_from_file_location(
        "codabench_submission", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowTests(unittest.TestCase):
    def test_windows_stop_at_first_n2(self):
        bounds = pre_n2_window_bounds(128 * 30, 128.0, 640, 12.0)
        ends = [end / 128.0 for _start, end in bounds]
        self.assertEqual(ends, [5.0, 10.0])
        exact = pre_n2_window_bounds(128 * 20, 128.0, 640, 10.0)
        exact_ends = [end / 128.0 for _start, end in exact]
        self.assertEqual(exact_ends, [5.0, 10.0])
        self.assertEqual(
            pre_n2_window_bounds(128 * 20, 128.0, 640, 4.0),
            [],
        )

    def test_target_is_remaining_time_clipped(self):
        self.assertEqual(remaining_seconds(10.0, 10.0), 0.0)
        self.assertEqual(remaining_seconds(50.0, 10.0), 40.0)
        self.assertEqual(remaining_seconds(1000.0, 10.0), 600.0)


class SplitTests(unittest.TestCase):
    def test_official_test_subjects_stay_in_training(self):
        subjects = np.array(["a", "a", "b", "c", "c", "d"])
        flags = np.array([1, 1, 0, 0, 0, 0])
        train_idx, val_idx = official_muse_split(
            subjects, flags, val_size=0.5, seed=33
        )
        assert_disjoint(subjects[train_idx], subjects[val_idx])
        self.assertIn("a", set(subjects[train_idx]))
        self.assertNotIn("a", set(subjects[val_idx]))
        self.assertTrue(set(subjects[val_idx]).issubset({"b", "c", "d"}))
        again_train, again_val = official_muse_split(
            subjects, flags, val_size=0.5, seed=33
        )
        self.assertTrue(np.array_equal(train_idx, again_train))
        self.assertTrue(np.array_equal(val_idx, again_val))

    def test_sessions_table_flags_a_test_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "sub-1"
            subject.mkdir()
            table = subject / "sub-1_sessions.tsv"
            table.write_text(
                "session_id\tsplit\n"
                "ses-a\ttrain\n"
                "ses-b\ttest\n",
                encoding="utf-8",
            )
            self.assertEqual(get_split(root, "sub-1", "ses-a"), "train")
            self.assertEqual(get_split(root, "sub-1", "ses-b"), "test")
            self.assertTrue(
                subject_has_official_test(root, "sub-1", "test")
            )


class MetricTests(unittest.TestCase):
    def test_recording_mean_matches_the_official_weights(self):
        frame = pd.DataFrame({
            "subject": ["s1", "s1", "s2"],
            "session": ["a", "a", "b"],
            "target_s": [10.0, 50.0, 600.0],
        })
        score = weighted_bmae(
            frame, np.array([12.0, 55.0, 500.0]), EDGES, WEIGHTS
        )
        self.assertAlmostEqual(score, 51.5)

    def test_last_bin_includes_the_cap(self):
        self.assertAlmostEqual(
            binned_mae([600.0], [500.0], EDGES, WEIGHTS), 100.0
        )
        self.assertTrue(
            np.isnan(binned_mae([600.1], [500.0], EDGES, WEIGHTS))
        )


class FeatureParityTests(unittest.TestCase):
    def test_training_and_submission_features_match(self):
        submission = load_submission()
        cfg = load_config("default")
        channels = cfg["data"]["channels"]
        bands = cfg["data"]["bands"]
        volts = np.random.default_rng(0).normal(
            scale=1e-5, size=(4, 640)
        )
        microvolts = volts * 1e6
        trained = training_features(microvolts, 128.0, channels, bands)
        streamed = submission.extract_features(
            microvolts, 128.0, channels, bands
        )
        self.assertEqual(list(trained), list(streamed))
        self.assertTrue(
            np.allclose(list(trained.values()), list(streamed.values()))
        )
        unscaled = training_features(volts, 128.0, channels, bands)
        self.assertFalse(
            np.allclose(list(trained.values()), list(unscaled.values()))
        )


class PortableModelTests(unittest.TestCase):
    def setUp(self):
        self.submission = load_submission()
        cfg = load_config("default")
        self.channels = cfg["data"]["channels"]
        self.bands = cfg["data"]["bands"]
        rng = np.random.default_rng(1)
        self.windows = rng.normal(scale=20.0, size=(12, 4, 640))
        rows = [
            training_features(window, 128.0, self.channels, self.bands)
            for window in self.windows
        ]
        self.frame = pd.DataFrame(rows)
        target = rng.uniform(0.0, 600.0, size=len(self.frame))
        self.model = LGBMRegressor(
            n_estimators=8,
            learning_rate=0.1,
            num_leaves=4,
            min_child_samples=1,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(self.frame, target)
        self.artifact = artifact_from_payload({
            "model": self.model,
            "features": list(self.frame.columns),
            "channels": self.channels,
            "bands": self.bands,
            "config": "synthetic",
            "split_seed": 33,
            "model_seed": 42,
            "sampling_hz": 128,
            "window_samples": 640,
        })

    def test_tree_walk_matches_lightgbm(self):
        walked = self.submission.predict_booster(
            self.artifact["booster"], self.frame.to_numpy()
        )
        self.assertTrue(
            np.allclose(walked, self.model.predict(self.frame), atol=1e-8)
        )
        missing = self.frame.iloc[[0]].copy()
        missing.iloc[0, 0] = np.nan
        walked_missing = self.submission.predict_booster(
            self.artifact["booster"], missing.to_numpy()
        )
        self.assertTrue(
            np.allclose(
                walked_missing,
                self.model.predict(missing),
                atol=1e-8,
                equal_nan=True,
            )
        )

    def test_predict_contract(self):
        portable = self.submission.MuseLightGBM(self.artifact)
        prediction = portable.predict(self.windows[:3])
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(prediction.dtype, np.float64)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.all(prediction >= 0.0))
        self.assertTrue(np.all(prediction <= 600.0))
        expected = np.clip(self.model.predict(self.frame.iloc[:3]), 0, 600)
        self.assertTrue(np.allclose(prediction, expected, atol=1e-8))
        reversed_windows = self.windows[:3][::-1]
        reversed_pred = portable.predict(reversed_windows)[::-1]
        self.assertTrue(np.allclose(prediction, reversed_pred, atol=1e-8))
        parameters = tuple(inspect.signature(portable.predict).parameters)
        self.assertEqual(parameters, ("batch",))

    def test_channel_order_follows_names(self):
        canonical = self.submission.MuseLightGBM(self.artifact)
        swapped = self.windows[:, [1, 0, 2, 3], :]
        renamed = self.submission.MuseLightGBM(
            self.artifact,
            ch_names=["AF7", "TP9", "AF8", "TP10"],
        )
        self.assertTrue(
            np.allclose(
                canonical.predict(self.windows[:2]),
                renamed.predict(swapped[:2]),
                atol=1e-8,
            )
        )

    def test_predictions_are_clipped_and_finite(self):
        high = json.loads(json.dumps(self.artifact))
        high["booster"] = {
            "tree_info": [{"tree_structure": {"leaf_value": 9000.0}}]
        }
        low = json.loads(json.dumps(self.artifact))
        low["booster"] = {
            "tree_info": [{"tree_structure": {"leaf_value": -5.0}}]
        }
        broken = json.loads(json.dumps(self.artifact))
        broken["booster"] = {
            "tree_info": [{"tree_structure": {"leaf_value": None}}]
        }
        window = self.windows[:1]
        self.assertEqual(
            self.submission.MuseLightGBM(high).predict(window)[0], 600.0
        )
        self.assertEqual(
            self.submission.MuseLightGBM(low).predict(window)[0], 0.0
        )
        fallback = self.submission.MuseLightGBM(broken).predict(window)
        self.assertEqual(fallback[0], 300.0)
        self.assertTrue(np.isfinite(fallback).all())

    def test_reset_and_deepcopy_keep_recordings_apart(self):
        model = self.submission.MuseLightGBM(self.artifact)
        model.predict(self.windows[:1])
        self.assertEqual(model._seen, 1)
        clone = copy.deepcopy(model)
        clone.reset_state()
        self.assertEqual(model._seen, 1)
        self.assertEqual(clone._seen, 0)
        self.assertTrue(
            np.allclose(
                model.predict(self.windows[:1]),
                clone.predict(self.windows[:1]),
                atol=1e-8,
            )
        )

    def test_cpu_tensor_and_solver_load(self):
        import torch

        portable = self.submission.MuseLightGBM(self.artifact)
        tensor = torch.tensor(self.windows[:1], dtype=torch.float32)
        self.assertEqual(tensor.device.type, "cpu")
        from_tensor = portable.predict(tensor)
        from_numpy = portable.predict(self.windows[:1])
        self.assertTrue(np.allclose(from_tensor, from_numpy, atol=1e-5))
        if torch.backends.mps.is_available():
            from_mps = portable.predict(tensor.to("mps"))
            self.assertTrue(np.allclose(from_tensor, from_mps, atol=1e-5))

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact_path = folder / "lightgbm_booster.json"
            artifact_path.write_text(
                json.dumps(self.artifact), encoding="utf-8"
            )
            loaded = self.submission.Solver().load_model({
                "submission_dir": folder,
                "sfreq": 128,
                "ch_names": self.channels,
                "device": "cpu",
            })
            self.assertTrue(
                np.allclose(
                    loaded.predict(self.windows[:1]),
                    from_numpy,
                    atol=1e-8,
                )
            )
            with self.assertRaises(ValueError):
                self.submission.Solver().load_model({
                    "submission_dir": folder,
                    "sfreq": 120,
                })
            empty = folder / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                self.submission.Solver().load_model({
                    "submission_dir": empty,
                    "sfreq": 128,
                })

    def test_zip_root_contains_only_the_submission_files(self):
        source = (
            REPO_ROOT / "muse_experiments" / "submission" / "submission.py"
        )
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("import lightgbm", text)
        self.assertNotIn("import joblib", text)
        self.assertNotIn("muse_experiments", text)
        requirements = (REPO_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("lightgbm", requirements)
        for name in ("benchopt", "pandas", "scikit-learn", "torch", "mne"):
            self.assertIn(name, requirements)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact_path = folder / "lightgbm_booster.json"
            artifact_path.write_text(
                json.dumps(self.artifact), encoding="utf-8"
            )
            zip_path = folder / "submission.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(source, arcname="submission.py")
                archive.write(
                    artifact_path, arcname="lightgbm_booster.json"
                )
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
            self.assertEqual(
                sorted(names),
                ["lightgbm_booster.json", "submission.py"],
            )
            self.assertTrue(all("/" not in name for name in names))


if __name__ == "__main__":
    unittest.main()
