"""Speed measurement, as defined in docs/benchmark-protocol.md.

The procedure (``run_speed``) and the arithmetic (``speed_metrics``) are kept
apart so the arithmetic can be tested with hand-made timings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ladder.backends.base import Backend, GenerationResult


@dataclass(frozen=True)
class Prompt:
    """One entry of the prompt set."""

    id: str
    text: str


@dataclass(frozen=True)
class Measurement:
    """One measured generation, tagged with the prompt it came from."""

    prompt_id: str
    result: GenerationResult


def load_prompts(path: str | Path) -> list[Prompt]:
    """Read a JSONL file with one ``{"id": ..., "text": ...}`` object per line."""
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prompts.append(Prompt(id=row["id"], text=row["text"]))
    return prompts


def token_hash(token_ids: list[int]) -> str:
    """SHA-256 of the token ids. Equal hashes mean identical generated text."""
    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()


def run_speed(
    backend: Backend,
    prompts: list[Prompt],
    max_new_tokens: int = 64,
    warmup: int = 3,
    repeats: int = 5,
) -> list[Measurement]:
    """Warm up, then generate for every prompt ``repeats`` times.

    Prompts are tokenized once, before any timing, so tokenizer speed is not
    part of the measurement. Warmup generations cycle through the prompts and
    are thrown away.
    """
    encoded = [(p.id, backend.encode(p.text)) for p in prompts]
    for i in range(warmup):
        _, ids = encoded[i % len(encoded)]
        backend.generate(ids, max_new_tokens)

    measurements = []
    for _ in range(repeats):
        for prompt_id, ids in encoded:
            result = backend.generate(ids, max_new_tokens)
            measurements.append(Measurement(prompt_id=prompt_id, result=result))
    return measurements


def speed_metrics(measurements: list[Measurement]) -> dict[str, float | None]:
    """Turn measured generations into the protocol's speed metrics.

    Times are reported in milliseconds. Inter-token gaps are pooled over all
    generations before taking percentiles; tokens/s figures are computed per
    generation and then the median is taken.
    """
    results = [m.result for m in measurements]
    ttft_ms = np.array([r.ttft_s for r in results]) * 1000
    itl_ms = np.array([gap for r in results for gap in r.inter_token_s]) * 1000
    decode_tps = [r.decode_tokens_per_s for r in results] if itl_ms.size else []
    e2e_tps = [len(r.token_ids) / r.total_s for r in results]

    metrics: dict[str, float | None] = {
        "ttft_ms_p50": float(np.percentile(ttft_ms, 50)),
        "ttft_ms_p95": float(np.percentile(ttft_ms, 95)),
        "decode_tps_median": float(np.median(decode_tps)) if decode_tps else None,
        "e2e_tps_median": float(np.median(e2e_tps)),
    }
    # With max_new_tokens == 1 there are no gaps between tokens; None, not NaN,
    # because NaN is not valid JSON.
    for p in (50, 95, 99):
        metrics[f"itl_ms_p{p}"] = float(np.percentile(itl_ms, p)) if itl_ms.size else None
    return metrics


def token_hashes(measurements: list[Measurement]) -> tuple[dict[str, str], bool]:
    """Hash of the generated tokens per prompt, and whether repeats agreed.

    Greedy decoding is deterministic, so every repeat of a prompt should give the
    same tokens. If it does not, the second value is False and the stored hash is
    the one from the first repeat.
    """
    hashes: dict[str, str] = {}
    consistent = True
    for m in measurements:
        h = token_hash(m.result.token_ids)
        if m.prompt_id not in hashes:
            hashes[m.prompt_id] = h
        elif hashes[m.prompt_id] != h:
            consistent = False
    return hashes, consistent
