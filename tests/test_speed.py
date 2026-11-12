import json
from pathlib import Path

import numpy as np
import pytest

from ladder.backends.base import Backend, BackendInfo, GenerationResult
from ladder.bench.speed import (
    Measurement,
    Prompt,
    load_prompts,
    run_speed,
    speed_metrics,
    token_hash,
    token_hashes,
)


class ScriptedBackend(Backend):
    """Returns pre-written timings, so the expected metrics are known exactly."""

    def __init__(self, scripts: dict[int, list[float]]) -> None:
        self.scripts = scripts  # first prompt token -> token_times_s
        self.calls: list[tuple[list[int], int]] = []

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("fake", "fake", "fake", "fp32", 0, 0.0, 1)

    def encode(self, text: str) -> list[int]:
        return [int(text)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(map(str, token_ids))

    def logits(self, input_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationResult:
        self.calls.append((prompt_ids, max_new_tokens))
        times = self.scripts[prompt_ids[0]]
        return GenerationResult(list(range(len(times))), times, len(prompt_ids))


# Prompt "1": gaps of 100, 200, 300 ms.  Prompt "2": gaps of 50, 100, 150 ms.
SCRIPTS = {1: [0.1, 0.2, 0.4, 0.7], 2: [0.2, 0.25, 0.35, 0.5]}
PROMPTS = [Prompt("a", "1"), Prompt("b", "2")]


def test_speed_metrics_known_answer():
    backend = ScriptedBackend(SCRIPTS)
    m = speed_metrics(run_speed(backend, PROMPTS, max_new_tokens=4, warmup=0, repeats=1))

    # TTFT values 100 and 200 ms; numpy's linear interpolation between them.
    assert m["ttft_ms_p50"] == pytest.approx(150)
    assert m["ttft_ms_p95"] == pytest.approx(195)
    # Pooled gaps, sorted: 50 100 100 150 200 300.
    assert m["itl_ms_p50"] == pytest.approx(125)
    assert m["itl_ms_p95"] == pytest.approx(275)
    assert m["itl_ms_p99"] == pytest.approx(295)
    # decode: 3 / 0.6 = 5 and 3 / 0.3 = 10 tok/s.  e2e: 4 / 0.7 and 4 / 0.5.
    assert m["decode_tps_median"] == pytest.approx(7.5)
    assert m["e2e_tps_median"] == pytest.approx((4 / 0.7 + 8) / 2)


def test_single_token_generation_has_no_inter_token_metrics():
    backend = ScriptedBackend({1: [0.1]})
    m = speed_metrics(run_speed(backend, [Prompt("a", "1")], 1, warmup=0, repeats=2))
    assert m["itl_ms_p50"] is None
    assert m["decode_tps_median"] is None
    assert m["ttft_ms_p50"] == pytest.approx(100)


def test_run_speed_does_warmup_then_repeats():
    backend = ScriptedBackend(SCRIPTS)
    measurements = run_speed(backend, PROMPTS, max_new_tokens=4, warmup=3, repeats=5)

    assert len(backend.calls) == 3 + 5 * len(PROMPTS)
    # Warmup cycles through the prompts and is not returned.
    assert [ids for ids, _ in backend.calls[:3]] == [[1], [2], [1]]
    assert len(measurements) == 5 * len(PROMPTS)
    assert [m.prompt_id for m in measurements[:2]] == ["a", "b"]
    assert all(n == 4 for _, n in backend.calls)


def test_token_hashes_per_prompt_and_consistency():
    same = GenerationResult([1, 2, 3], [0.1, 0.2, 0.3], 1)
    other = GenerationResult([1, 2, 4], [0.1, 0.2, 0.3], 1)

    hashes, consistent = token_hashes([Measurement("a", same), Measurement("a", same)])
    assert hashes == {"a": token_hash([1, 2, 3])}
    assert consistent

    _, consistent = token_hashes([Measurement("a", same), Measurement("a", other)])
    assert not consistent
    assert token_hash([1, 2, 3]) != token_hash([1, 2, 4])


def test_load_prompts(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "x", "text": "hello"}\n\n{"id": "y", "text": "world"}\n')
    assert load_prompts(path) == [Prompt("x", "hello"), Prompt("y", "world")]


def test_shipped_prompt_set_is_valid():
    path = Path(__file__).parents[1] / "data" / "prompts.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 16
    assert len({r["id"] for r in rows}) == 16
    assert all(r["text"].strip() for r in rows)


def test_run_speed_on_tiny_backend(tiny_backend):
    measurements = run_speed(tiny_backend, [Prompt("a", "The quick brown fox")], 8, 1, 2)
    m = speed_metrics(measurements)
    assert len(measurements) == 2
    assert m["decode_tps_median"] > 0
    assert m["ttft_ms_p50"] > 0
