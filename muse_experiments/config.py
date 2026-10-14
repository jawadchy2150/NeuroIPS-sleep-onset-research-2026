"""YAML configs for team experiments.

``device: auto`` resolves to CUDA, then Apple MPS, then CPU. Set
``device`` in a config, or override a run with ``MUSE_DEVICE``.
"""

import copy
import os
import re
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def default_muse_root():
    """NeuralBench Muse cache used by the current feature script."""
    override = os.environ.get("NEURALBENCH_MUSE_ROOT")
    if override:
        return Path(override).expanduser()
    return (
        Path.home()
        / ".neuralbench/data1/Interaxon2026Muse/download/nm000287"
    )


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand(value):
    if isinstance(value, str):
        return _ENV_RE.sub(
            lambda match: os.environ.get(match.group(1), ""),
            value,
        )
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def config_path(name):
    path = Path(name)
    if path.suffix in {".yml", ".yaml"}:
        return path if path.is_absolute() else Path.cwd() / path
    return CONFIG_DIR / f"{name}.yaml"


def load_config(name="default"):
    """Load a config, following ``extends`` from the configs directory."""
    path = config_path(name)
    seen = []
    loaded = _load_chain(path, seen)
    loaded = _expand(loaded)
    paths = loaded.setdefault("paths", {})
    if not paths.get("muse_data_root"):
        paths["muse_data_root"] = str(default_muse_root())
    loaded["config_path"] = str(path)
    return loaded


def _load_chain(path, seen):
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in seen)
        raise ValueError(f"config extend cycle at {resolved} ({chain})")
    seen.append(resolved)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    parent_name = raw.get("extends")
    if not parent_name:
        return raw
    parent = _load_chain(CONFIG_DIR / f"{parent_name}.yaml", seen)
    return _deep_merge(parent, raw)
