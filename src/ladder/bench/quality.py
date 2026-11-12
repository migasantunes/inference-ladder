"""Quality metrics, as defined in docs/benchmark-protocol.md.

* perplexity: how well the model predicts real text.
* top-1 agreement and KL divergence: how far a rung's predictions drifted from
  the reference (baseline) model's, position by position.

Everything works on arrays of token ids, so tests can use made-up data. The data
loaders at the bottom only turn a dataset or file into one long string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ladder.backends.base import Backend


def make_windows(
    token_ids: list[int] | np.ndarray, window: int, max_windows: int | None = None
) -> np.ndarray:
    """Cut a token stream into non-overlapping windows of exactly ``window`` tokens.

    The incomplete tail is dropped so every window has the same length.

    Returns:
        int64 array of shape ``(num_windows, window)``.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    num_windows = len(ids) // window
    if max_windows is not None:
        num_windows = min(num_windows, max_windows)
    if num_windows == 0:
        raise ValueError(f"Need at least {window} tokens for one window, got {len(ids)}")
    return ids[: num_windows * window].reshape(num_windows, window)


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    """Log-probabilities over the last axis, in float64.

    Subtracting the max before ``exp`` avoids overflow; float64 keeps the small
    differences that KL divergence is made of from being lost to rounding.
    """
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def kl_divergence(ref_logp: np.ndarray, cand_logp: np.ndarray) -> np.ndarray:
    """KL(reference || candidate) per position, in nats, from log-probabilities."""
    return (np.exp(ref_logp) * (ref_logp - cand_logp)).sum(axis=-1)


@dataclass(frozen=True)
class QualityResult:
    """Outcome of a quality run. Comparison fields are None without a reference."""

    perplexity: float
    top1_agreement: float | None
    mean_kl: float | None
    num_windows: int
    num_tokens: int


def evaluate_quality(
    backend: Backend, windows: np.ndarray, reference: Backend | None = None
) -> QualityResult:
    """Perplexity of ``backend`` and, optionally, its drift from ``reference``.

    Each window is one forward pass. For perplexity, position ``i`` predicts
    token ``i + 1``, so a window of W tokens gives W - 1 predictions. Agreement
    and KL compare the two models' predictions at all W positions.
    """
    nll_sum = 0.0
    nll_count = 0
    agree = 0
    kl_sum = 0.0
    positions = 0

    for row in windows:
        batch = row[None, :]  # shape (1, window)
        logp = log_softmax64(backend.logits(batch)[0])  # (window, vocab)
        targets = row[1:]
        nll_sum -= logp[np.arange(len(targets)), targets].sum()
        nll_count += len(targets)

        if reference is not None:
            ref_logp = log_softmax64(reference.logits(batch)[0])
            agree += int((logp.argmax(axis=-1) == ref_logp.argmax(axis=-1)).sum())
            kl_sum += float(kl_divergence(ref_logp, logp).sum())
            positions += len(row)

    return QualityResult(
        perplexity=math.exp(nll_sum / nll_count),
        top1_agreement=agree / positions if reference is not None else None,
        mean_kl=kl_sum / positions if reference is not None else None,
        num_windows=len(windows),
        num_tokens=int(windows.size),
    )


def load_wikitext2(split: str = "test") -> str:
    """WikiText-2 (raw) as one string, rows joined with blank lines.

    Needs the optional ``datasets`` package and network access the first time.
    """
    try:
        import datasets
    except ImportError as err:
        raise RuntimeError(
            "Loading WikiText-2 needs the 'datasets' package: "
            'pip install -e ".[data]". To run offline, pass --text-file PATH instead.'
        ) from err
    rows = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(rows["text"])


def load_text_file(path: str | Path) -> str:
    """A plain UTF-8 text file as one string (the offline alternative)."""
    return Path(path).read_text(encoding="utf-8")
