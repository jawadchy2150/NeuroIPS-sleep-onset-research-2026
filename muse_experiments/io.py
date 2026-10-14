"""Repo paths and writers that leave existing artifacts in place."""

import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PRESERVED_RELATIVE = (
    "muse_experiments/muse_features_100.csv",
    "muse_experiments/lightgbm_100.joblib",
    "muse_experiments/lightgbm_error_analysis.csv",
    "eeg_preview.png",
    "eeg_5s_before_n2.png",
)

_OFFICIAL_PARTS = {"data", "outputs", "__cache__"}


def repo_path(value):
    """Resolve a config path. Absolute paths stay absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def preserved_paths():
    return {repo_path(relative).resolve() for relative in PRESERVED_RELATIVE}


def _under_official_tree(path):
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    parts = relative.parts
    return bool(parts) and parts[0] == "tracks" and bool(
        _OFFICIAL_PARTS.intersection(parts)
    )


def allocate_output(desired):
    """Return a path that does not replace an existing file.

    Official ``tracks/*/data``, ``outputs``, and ``__cache__`` trees are
    refused. The feature table, LightGBM checkpoint, error table, and
    exploration figures already in the repo are sent to ``results/``.
    """
    path = repo_path(desired).resolve()
    if _under_official_tree(path):
        raise ValueError(
            "refusing to write inside official benchmark files: "
            f"{path}"
        )
    if path in preserved_paths():
        path = (
            REPO_ROOT
            / "muse_experiments"
            / "results"
            / "kept_aside"
            / Path(desired).name
        ).resolve()
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for index in range(1, 1000):
        suffix = path.suffix
        marker = stamp if index == 1 else f"{stamp}_{index}"
        candidate = path.with_name(f"{path.stem}__{marker}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no free output name for {path}")


def publish_file(desired, write_temp):
    """Write via ``write_temp(tmp_path)``, then link the result into place.

    ``write_temp`` must create ``tmp_path``. An existing destination is
    never replaced.
    """
    last_error = None
    for _attempt in range(20):
        path = allocate_output(desired)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".partial-{path.name}")
        write_temp(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            continue
        temporary.unlink()
        return path
    raise FileExistsError(
        f"could not publish a new file for {desired}"
    ) from last_error
