"""Write a fitted LightGBM model as JSON for the Codabench worker.

The worker image does not install LightGBM. ``Booster.dump_model()``
stores each leaf already scaled by that tree's shrinkage, so a portable
walker adds the leaf values and matches ``Booster.predict``.
"""

import json

from muse_experiments.io import publish_file


FORMAT = "lightgbm-dump-model-v1"


def artifact_from_payload(payload):
    """JSON-ready record of one fitted payload."""
    model = payload["model"]
    booster = model.booster_.dump_model()
    features = list(payload["features"])
    dumped_names = list(booster.get("feature_names") or [])
    if dumped_names and dumped_names != features:
        raise ValueError(
            "LightGBM feature order does not match the saved names"
        )
    bands = {
        name: [float(low), float(high)]
        for name, (low, high) in payload["bands"].items()
    }
    return {
        "format": FORMAT,
        "config": payload.get("config"),
        "split_seed": payload.get("split_seed"),
        "model_seed": payload.get("model_seed"),
        "features": features,
        "channels": list(payload["channels"]),
        "bands": bands,
        "sampling_hz": float(payload["sampling_hz"]),
        "window_samples": int(payload["window_samples"]),
        "cap_seconds": 600.0,
        "fallback_seconds": 300.0,
        "booster": booster,
    }


def write_booster_json(payload, model_path):
    """Write ``<model stem>_booster.json`` beside a saved joblib file."""
    if not hasattr(payload.get("model"), "booster_"):
        return None
    artifact = artifact_from_payload(payload)
    desired = model_path.with_name(model_path.stem + "_booster.json")

    def write_temp(temporary):
        temporary.write_text(json.dumps(artifact), encoding="utf-8")

    return publish_file(desired, write_temp)
