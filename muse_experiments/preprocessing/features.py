"""Window features for the Muse warm-up stream.

``extract_features`` expects microvolts. Offline BrainVision files are in
volts, so the preparation script multiplies by ``1e6`` once before calling
it. Codabench already passes microvolts, and must not scale again.
"""

import numpy as np
from scipy.signal import welch

METADATA_COLUMNS = [
    "subject",
    "session",
    "window_start_s",
    "window_end_s",
    "target_s",
    "has_official_test",
]


def feature_names(bands):
    return ["std_uv", "line_length"] + [
        f"{band}_power" for band in bands
    ]


def feature_columns(channels, bands):
    names = feature_names(bands)
    return [
        f"{channel}_{feature}"
        for channel in channels
        for feature in names
    ]


def remaining_seconds(n2_onset_s, window_end_s, cap_seconds=600.0):
    """Seconds from the window end until first N2, clipped to ``0–cap``."""
    return float(np.clip(n2_onset_s - window_end_s, 0.0, cap_seconds))


def pre_n2_window_bounds(n_times, sfreq, window_samples, n2_onset_s):
    """Sample bounds of 5 s windows that end at or before first N2.

    A window that would cross or follow N2 is omitted. This matches the
    NeuralBench 0.4.0 stream: non-overlapping segments inside the marker
    that stops at the first N2 onset.
    """
    bounds = []
    if window_samples <= 0 or n_times < window_samples:
        return bounds
    for start in range(0, n_times - window_samples + 1, window_samples):
        end = start + window_samples
        if (end / sfreq) > n2_onset_s + 1e-9:
            break
        bounds.append((start, end))
    return bounds


def extract_features(data_uv, sfreq, channels, bands):
    """Features for one window. ``data_uv`` is microvolts, shape ``(C, T)``."""
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
