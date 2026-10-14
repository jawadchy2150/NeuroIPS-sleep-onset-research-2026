"""Window features shared by Muse preprocessing and the feature-table tests."""

import numpy as np
from scipy.signal import welch

METADATA_COLUMNS = [
    "subject",
    "session",
    "window_start_s",
    "window_end_s",
    "target_s",
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


def extract_features(data, sfreq, channels, bands):
    """Features for one window. ``data`` is volts, shape ``(C, T)``."""
    features = {}
    data_uv = data * 1e6
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
