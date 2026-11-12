"""The contract every rung implements.

A *backend* is one way of running the same language model: plain PyTorch, ONNX
Runtime, ONNX Runtime with int8 weights, and so on. The benchmark harness and the
HTTP service only ever talk to this interface, which is what makes the rungs
comparable: same prompts, same decoding, same measurements, different engine.

Two operations are enough for everything we measure:

* ``logits`` - a full forward pass over a batch of token ids. Used by the quality
  metrics (perplexity, agreement with the baseline), where we need the model's
  raw predictions at every position.
* ``generate`` - greedy autoregressive decoding of one prompt, reporting the wall
  clock time at which each new token became available. Used by the speed metrics.

Decoding is always greedy (argmax). Sampling would add randomness to the timings
and make two rungs produce different text for reasons unrelated to the
optimization being measured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of one greedy generation.

    Attributes:
        token_ids: The newly generated token ids only (the prompt is excluded).
        token_times_s: For each generated token, seconds elapsed since
            ``generate`` was called when that token became available. Monotonic
            and non-decreasing; ``token_times_s[0]`` is the time to first token.
        prompt_tokens: Number of tokens in the prompt.
    """

    token_ids: list[int]
    token_times_s: list[float]
    prompt_tokens: int

    @property
    def ttft_s(self) -> float:
        """Time to first token, in seconds."""
        return self.token_times_s[0]

    @property
    def total_s(self) -> float:
        """Wall clock time for the whole generation, in seconds."""
        return self.token_times_s[-1]

    @property
    def inter_token_s(self) -> list[float]:
        """Gaps between consecutive tokens, excluding the first token."""
        t = self.token_times_s
        return [b - a for a, b in zip(t[:-1], t[1:], strict=True)]

    @property
    def decode_tokens_per_s(self) -> float:
        """Steady-state decoding speed, excluding the first token.

        The first token pays for processing the whole prompt, so it is reported
        separately as TTFT and left out of the throughput figure.
        """
        if len(self.token_times_s) < 2:
            return float("nan")
        return (len(self.token_times_s) - 1) / (self.total_s - self.ttft_s)


@dataclass(frozen=True)
class BackendInfo:
    """Static facts about a loaded backend, recorded with every result row."""

    rung: str
    model_id: str
    engine: str
    precision: str
    size_on_disk_bytes: int
    load_time_s: float
    num_threads: int
    extra: dict[str, str] = field(default_factory=dict)


class Backend(ABC):
    """One way of running the model. Implement this to add a rung."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Static description of this backend."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` into ids, without adding padding."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Turn token ids back into text."""

    @abstractmethod
    def logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Full forward pass.

        Args:
            input_ids: int64 array of shape ``(batch, seq_len)``. All rows have
                the same length; callers do not pass padding.

        Returns:
            float32 array of shape ``(batch, seq_len, vocab_size)``.
        """

    @abstractmethod
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationResult:
        """Greedy-decode exactly ``max_new_tokens`` tokens after ``prompt_ids``.

        Implementations must not stop early on an end-of-sequence token: every
        rung has to produce the same number of tokens for timings to compare.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Release resources. Optional."""
