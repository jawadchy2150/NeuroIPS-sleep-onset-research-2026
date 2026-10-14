"""Checks for team config, device fallback, and subject-disjoint splits."""

import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from muse_experiments.config import load_config
from muse_experiments.device import resolve_device
from muse_experiments.evaluation.metrics import (
    bin_sample_weights,
    weighted_bmae,
)
from muse_experiments.io import (
    REPO_ROOT,
    allocate_output,
    preserved_paths,
    publish_file,
    repo_path,
)
from muse_experiments.preprocessing.features import (
    METADATA_COLUMNS,
    extract_features,
    feature_columns,
)
from muse_experiments.splits import assert_disjoint, participant_split
from muse_experiments.training.lightgbm_run import fit_lightgbm, save_model


class ConfigTests(unittest.TestCase):
    def test_lightgbm_config_matches_current_script(self):
        cfg = load_config("lightgbm_weighted")
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg["model"]["n_estimators"], 300)
        self.assertEqual(cfg["model"]["learning_rate"], 0.03)
        self.assertEqual(cfg["model"]["num_leaves"], 15)
        self.assertTrue(cfg["training"]["sample_weight"])
        self.assertEqual(cfg["split"]["group_column"], "subject")
        self.assertEqual(cfg["split"]["val_size"], 0.20)
        self.assertTrue(
            cfg["outputs"]["model_file"].endswith(
                "results/models/lightgbm_weighted_100.joblib"
            )
        )
        self.assertFalse(
            cfg["outputs"]["model_file"].endswith(
                "muse_experiments/lightgbm_100.joblib"
            )
        )
        self.assertEqual(
            cfg["data"]["channels"], ["TP9", "AF7", "AF8", "TP10"]
        )
        self.assertTrue(cfg["paths"]["muse_data_root"])

    def test_unweighted_config_disables_sample_weight(self):
        cfg = load_config("lightgbm_100")
        self.assertFalse(cfg["training"]["sample_weight"])
        self.assertEqual(cfg["name"], "lightgbm_100")


class DeviceTests(unittest.TestCase):
    def test_cpu_request_stays_on_cpu(self):
        choice = resolve_device("cpu")
        self.assertEqual(choice.name, "cpu")
        self.assertIsNone(choice.fallback_from)

    def test_auto_selects_an_available_device(self):
        choice = resolve_device("auto")
        self.assertIn(choice.name, {"cpu", "mps", "cuda"})
        self.assertEqual(choice.device.type, choice.name)

    def test_missing_cuda_falls_back_on_this_machine(self):
        choice = resolve_device("cuda")
        if choice.name == "cuda":
            self.assertIsNone(choice.fallback_from)
        else:
            self.assertEqual(choice.fallback_from, "cuda")
            self.assertIn(choice.name, {"mps", "cpu"})

    def test_missing_accelerator_can_refuse_fallback(self):
        choice = resolve_device("cuda")
        if choice.name == "cuda":
            self.skipTest("CUDA is available")
        with self.assertRaises(RuntimeError):
            resolve_device("cuda", fallback=False)

    def test_unknown_device_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu")


class SplitTests(unittest.TestCase):
    def test_participant_split_keeps_subjects_apart(self):
        subjects = np.repeat([f"s{i}" for i in range(10)], 4)
        train_idx, val_idx = participant_split(
            subjects, val_size=0.2, seed=42
        )
        assert_disjoint(subjects[train_idx], subjects[val_idx])
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        self.assertEqual(len(train_idx) + len(val_idx), len(subjects))

        again_train, again_val = participant_split(
            subjects, val_size=0.2, seed=42
        )
        self.assertTrue(np.array_equal(train_idx, again_train))
        self.assertTrue(np.array_equal(val_idx, again_val))

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_disjoint(["a", "b"], ["b", "c"])


class LayoutTests(unittest.TestCase):
    def test_model_outputs_are_distinct(self):
        plain = load_config("lightgbm_100")
        weighted = load_config("lightgbm_weighted")
        plain_out = repo_path(plain["outputs"]["model_file"]).resolve()
        weighted_out = repo_path(weighted["outputs"]["model_file"]).resolve()
        preserved = repo_path(plain["inputs"]["model_file"]).resolve()
        self.assertNotEqual(plain_out, weighted_out)
        self.assertNotEqual(plain_out, preserved)
        self.assertNotEqual(weighted_out, preserved)
        self.assertTrue(preserved.is_file())

    def test_feature_and_analysis_outputs_spare_existing_files(self):
        cfg = load_config("lightgbm_100")
        existing_csv = repo_path(cfg["inputs"]["features_csv"]).resolve()
        new_csv = repo_path(cfg["outputs"]["features_csv"]).resolve()
        analysis = repo_path(cfg["outputs"]["analysis_csv"]).resolve()
        old_analysis = (
            REPO_ROOT / "muse_experiments" / "lightgbm_error_analysis.csv"
        ).resolve()
        self.assertNotEqual(existing_csv, new_csv)
        self.assertNotEqual(analysis, old_analysis)
        self.assertTrue(existing_csv.is_file())
        self.assertTrue(old_analysis.is_file())

    def test_allocate_output_does_not_touch_preserved_files(self):
        before = {
            path: path.stat().st_mtime_ns for path in preserved_paths()
        }
        for path in preserved_paths():
            allocated = allocate_output(path)
            self.assertNotEqual(allocated, path)
            self.assertFalse(allocated.exists())
        after = {
            path: path.stat().st_mtime_ns for path in preserved_paths()
        }
        self.assertEqual(before, after)

    def test_official_benchmark_trees_are_refused(self):
        with self.assertRaises(ValueError):
            allocate_output("tracks/sleep_onset/outputs/new_score.parquet")
        with self.assertRaises(ValueError):
            allocate_output("tracks/sleep_onset/data/extra.csv")

    def test_second_publish_keeps_the_first_file(self):
        with tempfile.TemporaryDirectory() as directory:
            desired = Path(directory) / "model.joblib"
            first = publish_file(
                desired, lambda temporary: temporary.write_text("one")
            )
            second = publish_file(
                desired, lambda temporary: temporary.write_text("two")
            )
            self.assertEqual(first.read_text(), "one")
            self.assertEqual(second.read_text(), "two")
            self.assertNotEqual(first, second)

    def test_official_get_device_has_no_mps_branch(self):
        source = (REPO_ROOT / "benchmark_utils" / "data.py").read_text()
        start = source.index("def get_device")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        self.assertIn("COMPET_DEVICE", body)
        self.assertNotIn("mps", body)


class FeatureTests(unittest.TestCase):
    def test_feature_names_match_existing_table(self):
        cfg = load_config("preprocess_muse_100")
        expected = feature_columns(
            cfg["data"]["channels"], cfg["data"]["bands"]
        )
        header = pd.read_csv(
            repo_path(cfg["inputs"]["features_csv"]), nrows=0
        )
        self.assertEqual(
            list(header.columns),
            METADATA_COLUMNS + expected,
        )

    def test_extract_features_uses_configured_names(self):
        cfg = load_config("default")
        channels = cfg["data"]["channels"]
        bands = cfg["data"]["bands"]
        data = np.random.default_rng(0).normal(size=(len(channels), 128))
        extracted = extract_features(data, 256.0, channels, bands)
        self.assertEqual(
            list(extracted),
            feature_columns(channels, bands),
        )

    def test_saved_model_features_exist_in_the_table(self):
        cfg = load_config("lightgbm_100")
        header = pd.read_csv(
            repo_path(cfg["inputs"]["features_csv"]), nrows=0
        )
        saved = joblib.load(repo_path(cfg["inputs"]["model_file"]))
        missing = [
            name for name in saved["features"] if name not in header.columns
        ]
        self.assertEqual(missing, [])
        self.assertIn("model", saved)


class MetricTests(unittest.TestCase):
    def test_sample_weights_follow_the_original_formula(self):
        frame = pd.DataFrame({
            "subject": ["s1", "s1", "s2"],
            "session": ["a", "a", "a"],
            "target_s": [10.0, 50.0, 100.0],
        })
        weights = bin_sample_weights(
            frame, [0, 40, 90, 300, 600], [10, 5, 3, 1]
        )
        expected = np.array([10.0, 5.0, 3.0])
        expected = expected / expected.mean()
        self.assertTrue(np.allclose(weights, expected))

    def test_weighted_bmae_matches_a_single_recording(self):
        frame = pd.DataFrame({
            "subject": ["s1", "s1"],
            "session": ["a", "a"],
            "target_s": [10.0, 50.0],
        })
        score = weighted_bmae(
            frame,
            np.array([12.0, 55.0]),
            [0, 40, 90, 300, 600],
            [10, 5, 3, 1],
        )
        # Bins 0-40 and 40-90, weights 10 and 5. Errors are 2 and 5.
        hand = (2.0 * 10 + 5.0 * 5) / 15
        self.assertAlmostEqual(score, hand)


class TrainingSmokeTests(unittest.TestCase):
    def test_synthetic_fit_writes_only_the_requested_file(self):
        preserved = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in preserved_paths()
        }
        cfg = copy.deepcopy(load_config("lightgbm_weighted"))
        cfg["model"]["n_estimators"] = 3
        cfg["model"]["min_child_samples"] = 1
        cfg["split"]["val_size"] = 0.5
        frame = pd.DataFrame({
            "subject": ["s1", "s1", "s2", "s2"],
            "session": ["a", "a", "b", "b"],
            "window_start_s": [0, 5, 0, 5],
            "window_end_s": [5, 10, 5, 10],
            "target_s": [10.0, 50.0, 100.0, 400.0],
            "TP9_std_uv": [0.1, 0.2, 0.3, 0.4],
            "AF7_std_uv": [1.0, 1.0, 1.0, 1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "weighted.joblib"
            payload = fit_lightgbm(frame, cfg)
            written = save_model(payload, destination)
            self.assertEqual(written, destination.resolve())
            loaded = joblib.load(written)
            self.assertEqual(loaded["config"], "lightgbm_weighted")
            self.assertIn(loaded["device"], {"cpu", "mps", "cuda"})
        after = {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in preserved_paths()
        }
        self.assertEqual(preserved, after)

    def test_reorganized_modules_import(self):
        import explore_eeg
        import explore_muse
        import prepare_muse_features
        import muse_experiments.analyze_lightgbm as analyze_wrapper
        import muse_experiments.evaluation.analyze_lightgbm
        import muse_experiments.preprocessing.explore_eeg
        import muse_experiments.preprocessing.explore_muse
        import muse_experiments.preprocessing.prepare_muse_features
        import muse_experiments.train_lightgbm as train_wrapper
        import muse_experiments.train_lightgbm_weighted as weighted_wrapper
        import muse_experiments.training.train_lightgbm
        import muse_experiments.training.train_lightgbm_weighted

        self.assertTrue(callable(train_wrapper.main))
        self.assertTrue(callable(weighted_wrapper.main))
        self.assertTrue(callable(analyze_wrapper.main))
        self.assertTrue(callable(prepare_muse_features.main))
        self.assertTrue(callable(explore_eeg.main))
        self.assertTrue(callable(explore_muse.main))
        self.assertTrue(
            callable(muse_experiments.training.train_lightgbm.main)
        )


if __name__ == "__main__":
    unittest.main()
