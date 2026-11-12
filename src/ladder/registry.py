"""Rung names mapped to the code that builds each backend.

Each factory imports its engine inside the function, so listing rungs or
importing ``ladder`` never pulls in torch, ONNX Runtime or anything else heavy.
"""

from __future__ import annotations

from collections.abc import Callable

from ladder.backends.base import Backend

BASELINE_RUNG = "01_torch_fp32"


def _torch_nocache(model_id: str, num_threads: int | None) -> Backend:
    from ladder.backends.torch_backend import TorchBackend

    return TorchBackend(model_id, rung="00_torch_nocache", use_cache=False, num_threads=num_threads)


def _torch_fp32(model_id: str, num_threads: int | None) -> Backend:
    from ladder.backends.torch_backend import TorchBackend

    return TorchBackend(model_id, rung="01_torch_fp32", use_cache=True, num_threads=num_threads)


_FACTORIES: dict[str, Callable[[str, int | None], Backend]] = {
    "00_torch_nocache": _torch_nocache,
    "01_torch_fp32": _torch_fp32,
}


def available_rungs() -> list[str]:
    """Names of all registered rungs, sorted."""
    return sorted(_FACTORIES)


def build_backend(rung: str, model_id: str, num_threads: int | None) -> Backend:
    """Construct the backend for ``rung``.

    ``model_id`` is a Hugging Face hub id or a local model directory.
    """
    if rung not in _FACTORIES:
        raise ValueError(f"Unknown rung {rung!r}. Available rungs: {', '.join(available_rungs())}")
    return _FACTORIES[rung](model_id, num_threads)
