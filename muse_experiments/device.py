"""Device selection for team experiments.

Official Benchopt code keeps using ``benchmark_utils.data.get_device``
(``COMPET_DEVICE``, otherwise CUDA or CPU). Call ``resolve_device`` from
team training code when a run should also be able to use Apple Silicon.
"""

import os
from dataclasses import dataclass

_CHOICES = ("auto", "cpu", "mps", "cuda")


@dataclass(frozen=True)
class DeviceChoice:
    """Resolved torch device and whether a fallback was required."""

    device: object
    requested: str
    name: str
    fallback_from: str | None = None


def _preference(preference):
    if preference is None or preference == "":
        preference = (
            os.environ.get("MUSE_DEVICE")
            or os.environ.get("COMPET_DEVICE")
            or "auto"
        )
    preference = str(preference).strip().lower()
    if preference not in _CHOICES:
        allowed = ", ".join(_CHOICES)
        raise ValueError(
            f"device must be one of {allowed}, got {preference!r}"
        )
    return preference


def _availability():
    import torch

    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    )
    return {
        "cuda": bool(torch.cuda.is_available()),
        "mps": mps,
        "cpu": True,
    }


def describe_device(choice):
    """Short label for logs. Includes a fallback note when one happened."""
    if choice.fallback_from:
        return (
            f"{choice.name} (requested {choice.fallback_from}, "
            "using the next available device)"
        )
    return choice.name


def resolve_device(preference=None, *, fallback=True):
    """Pick a torch device.

    ``preference`` is ``auto``, ``cpu``, ``mps``, or ``cuda``. When it is
    omitted, ``MUSE_DEVICE`` is used, then ``COMPET_DEVICE``, then ``auto``.

    ``auto`` tries CUDA, then MPS, then CPU. A named accelerator that is
    missing falls through that same order when ``fallback`` is true.
    """
    import torch

    requested = _preference(preference)
    available = _availability()
    order = ("cuda", "mps", "cpu")
    if requested == "auto":
        candidates = order
    else:
        rest = tuple(name for name in order if name != requested)
        candidates = (requested,) + rest

    chosen = None
    for name in candidates:
        if available[name]:
            chosen = name
            break
        if requested != "auto" and not fallback:
            break

    if chosen is None:
        raise RuntimeError(
            f"requested device {requested!r} is unavailable"
        )

    used_fallback = requested != "auto" and chosen != requested
    return DeviceChoice(
        device=torch.device(chosen),
        requested=requested,
        name=chosen,
        fallback_from=requested if used_fallback else None,
    )
