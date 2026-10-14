"""Codabench submission for the corrected Muse LightGBM model.

All inference Python lives in this file. The worker image does not
install LightGBM, so predictions walk a JSON tree dump. Leaf values from
``Booster.dump_model()`` are already scaled; adding them matches
``Booster.predict``.

Place ``lightgbm_booster.json`` next to this file in the ZIP. Training
writes a longer unique name. Copy the chosen file to this name when
packaging. Do not ship the earlier joblib checkpoint.
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0
ARTIFACT = "lightgbm_booster.json"


def extract_features(data_uv, sfreq, channels, bands):
    """One window in microvolts, shape ``(C, T)``. Same math as training."""
    features = {}
    data_uv = np.asarray(data_uv, dtype=np.float64)
    freqs, psd = welch(
        data_uv,
        fs=sfreq,
        nperseg=min(256, data_uv.shape[1]),
        axis=1,
    )
    total_mask = (freqs >= 0.5) & (freqs < 30)

    for index, channel in enumerate(channels):
        signal = data_uv[index]
        power = psd[index]
        features[f"{channel}_std_uv"] = float(np.std(signal))
        features[f"{channel}_line_length"] = float(
            np.mean(np.abs(np.diff(signal)))
        )
        total_power = np.sum(power[total_mask]) + 1e-12
        for band, bounds in bands.items():
            low, high = bounds
            mask = (freqs >= low) & (freqs < high)
            features[f"{channel}_{band}_power"] = float(
                np.sum(power[mask]) / total_power
            )
    return features


def _leaf(node, row):
    if "leaf_value" in node:
        value = node["leaf_value"]
        if value is None:
            return float("nan")
        return float(value)
    decision = node.get("decision_type")
    if decision != "<=":
        raise ValueError(f"unsupported LightGBM decision {decision!r}")
    value = row[int(node["split_feature"])]
    if not np.isfinite(value):
        go_left = bool(node.get("default_left", True))
    else:
        go_left = float(value) <= float(node["threshold"])
    child = "left_child" if go_left else "right_child"
    return _leaf(node[child], row)


def predict_booster(booster, features):
    """Sum dumped leaf values for each row."""
    rows = np.asarray(features, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    scores = np.zeros(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        total = 0.0
        for tree in booster["tree_info"]:
            total += _leaf(tree["tree_structure"], row)
        scores[row_index] = total
    return scores


def _as_numpy(batch):
    if hasattr(batch, "detach"):
        batch = batch.detach().cpu().numpy()
    return np.asarray(batch, dtype=np.float64)


class MuseLightGBM:
    """Current-window EEG features and a portable LightGBM walk."""

    def __init__(self, artifact, ch_names=None):
        if artifact.get("format") != "lightgbm-dump-model-v1":
            raise ValueError("not a lightgbm-dump-model-v1 artifact")
        self.channels = list(artifact["channels"])
        self.bands = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in artifact["bands"].items()
        }
        self.feature_names = list(artifact["features"])
        self.sfreq = float(artifact["sampling_hz"])
        self.cap = float(artifact.get("cap_seconds", CAP_S))
        self.fallback = float(artifact.get("fallback_seconds", 300.0))
        self.booster = artifact["booster"]
        if ch_names is None:
            self.ch_names = None
        else:
            self.ch_names = [str(name) for name in ch_names]
        self._seen = 0

    def reset_state(self):
        """Clear per-recording state. Features do not read it."""
        self._seen = 0

    def _order(self, data):
        if data.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {data.shape}")
        if self.ch_names is None:
            if data.shape[1] != len(self.channels):
                raise ValueError(
                    f"expected {len(self.channels)} channels, "
                    f"got {data.shape[1]}"
                )
            return data
        index = [self.ch_names.index(channel) for channel in self.channels]
        return data[:, index, :]

    def _vector(self, window):
        extracted = extract_features(
            window, self.sfreq, self.channels, self.bands
        )
        return np.array(
            [extracted[name] for name in self.feature_names],
            dtype=np.float64,
        )

    def predict(self, batch):
        data = self._order(_as_numpy(batch))
        self._seen += int(data.shape[0])
        vectors = np.stack([self._vector(window) for window in data])
        scores = predict_booster(self.booster, vectors)
        scores = np.clip(scores, 0.0, self.cap)
        invalid = ~np.isfinite(scores)
        if np.any(invalid):
            scores = np.where(invalid, self.fallback, scores)
        return np.asarray(scores, dtype=np.float64)


class Solver(CompetSolver):

    name = "Muse-LightGBM"

    def load_model(self, meta):
        path = Path(meta["submission_dir"]) / ARTIFACT
        if not path.is_file():
            raise FileNotFoundError(
                f"{ARTIFACT} is missing from {path.parent}. Train the "
                "corrected model and copy its booster JSON to this name."
            )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        meta_sfreq = meta.get("sfreq")
        expected = float(artifact["sampling_hz"])
        if (
            meta_sfreq is not None
            and abs(float(meta_sfreq) - expected) > 1e-6
        ):
            raise ValueError(
                f"submission expects {expected} Hz, meta says {meta_sfreq}"
            )
        return MuseLightGBM(artifact, ch_names=meta.get("ch_names"))
