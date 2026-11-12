"""Tests for the HTTP service, run against a fake backend.

The fake backend needs no model, no torch and no network, so these tests check
the service's own behaviour: routing, validation, the response fields, and that
the single worker really serialises calls into the backend.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ladder.backends.base import Backend, BackendInfo, GenerationResult
from ladder.serve.app import create_app

# Scripted timings: first token after 10 ms, then one token every 5 ms.
TTFT_S = 0.010
STEP_S = 0.005


class FakeBackend(Backend):
    """Deterministic stand-in for a real backend.

    Encoding is one id per whitespace-separated word; generated token ``i`` is
    ``i % 26`` and decodes to the letter ``"a" + i``. It also records how many
    threads are inside ``generate`` at once, so a test can detect overlap.
    """

    def __init__(self, work_s: float = 0.0) -> None:
        self.work_s = work_s
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0
        # Tests can clear this to make generate() block until they set it again.
        self.proceed = threading.Event()
        self.proceed.set()
        self.entered = threading.Event()

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            rung="fake",
            model_id="fake/model",
            engine="fake",
            precision="fp32",
            size_on_disk_bytes=1234,
            load_time_s=0.5,
            num_threads=2,
            extra={"note": "test"},
        )

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(ord("a") + i % 26) for i in token_ids)

    def logits(self, input_ids: np.ndarray) -> np.ndarray:
        batch, seq_len = input_ids.shape
        return np.zeros((batch, seq_len, 26), dtype=np.float32)

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> GenerationResult:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            self.proceed.wait(timeout=5)
            time.sleep(self.work_s)
            return GenerationResult(
                token_ids=[i % 26 for i in range(max_new_tokens)],
                token_times_s=[TTFT_S + STEP_S * i for i in range(max_new_tokens)],
                prompt_tokens=len(prompt_ids),
            )
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(backend: FakeBackend):
    # Using TestClient as a context manager runs the app's startup and shutdown.
    with TestClient(create_app(backend)) as c:
        yield c


def test_healthz_ok_after_startup(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_503_before_startup(backend: FakeBackend) -> None:
    # Without the context manager the startup hook never runs: nothing is loaded.
    c = TestClient(create_app(backend))
    assert c.get("/healthz").status_code == 503
    assert c.post("/generate", json={"prompt": "hi"}).status_code == 503


def test_info_returns_backend_info(client: TestClient) -> None:
    resp = client.get("/info")
    assert resp.status_code == 200
    assert resp.json() == {
        "rung": "fake",
        "model_id": "fake/model",
        "engine": "fake",
        "precision": "fp32",
        "size_on_disk_bytes": 1234,
        "load_time_s": 0.5,
        "num_threads": 2,
        "extra": {"note": "test"},
    }


def test_generate_happy_path(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": "the quick brown fox", "max_new_tokens": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "abcde"
    assert body["prompt_tokens"] == 4
    assert body["generated_tokens"] == 5
    assert body["ttft_ms"] == pytest.approx(10.0)
    assert body["total_ms"] == pytest.approx(30.0)  # 10 ms + 4 steps of 5 ms


def test_generate_default_max_new_tokens(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": "hello"})
    assert resp.status_code == 200
    assert resp.json()["generated_tokens"] == 64


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hello", "max_new_tokens": 0},
        {"prompt": "hello", "max_new_tokens": 257},
        {"max_new_tokens": 5},
    ],
    ids=["empty-prompt", "zero-tokens", "too-many-tokens", "missing-prompt"],
)
def test_generate_validation_errors(
    client: TestClient, backend: FakeBackend, payload: dict[str, object]
) -> None:
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 422
    assert backend.calls == 0  # rejected before reaching the model


def test_max_new_tokens_upper_bound_is_allowed(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 256})
    assert resp.status_code == 200
    assert resp.json()["generated_tokens"] == 256


def test_concurrent_requests_never_overlap_in_backend() -> None:
    # Each generation takes a little while, so without the single worker the
    # requests fired from several threads would be inside generate() together.
    fake = FakeBackend(work_s=0.02)
    n_requests = 8
    with TestClient(create_app(fake)) as c:

        def call(i: int) -> int:
            resp = c.post("/generate", json={"prompt": f"prompt {i}", "max_new_tokens": 3})
            return resp.status_code

        with ThreadPoolExecutor(max_workers=n_requests) as pool:
            statuses = list(pool.map(call, range(n_requests)))

    assert statuses == [200] * n_requests
    assert fake.calls == n_requests
    assert fake.max_active == 1


def test_startup_builds_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub ladder.registry so this test never needs the real model code.
    seen: dict[str, object] = {}

    def fake_build_backend(rung: str, model_id: str, num_threads: int | None) -> Backend:
        seen.update(rung=rung, model_id=model_id, num_threads=num_threads)
        return FakeBackend()

    monkeypatch.setitem(
        sys.modules, "ladder.registry", types.SimpleNamespace(build_backend=fake_build_backend)
    )
    monkeypatch.setenv("LADDER_RUNG", "02_ort_fp32")
    monkeypatch.setenv("LADDER_MODEL", "some/model")
    monkeypatch.setenv("LADDER_THREADS", "3")
    with TestClient(create_app()) as c:
        assert c.get("/healthz").status_code == 200
    assert seen == {"rung": "02_ort_fp32", "model_id": "some/model", "num_threads": 3}


def test_bad_threads_env_var_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "ladder.registry", types.SimpleNamespace(build_backend=lambda **_: None)
    )
    monkeypatch.setenv("LADDER_THREADS", "two")
    with pytest.raises(ValueError, match="LADDER_THREADS"), TestClient(create_app()):
        pass


def test_healthz_responsive_while_generating() -> None:
    # Block the backend mid-generation; /healthz must still answer because the
    # generation runs on the worker thread, not on the event loop.
    fake = FakeBackend()
    fake.proceed.clear()
    with TestClient(create_app(fake)) as c, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(c.post, "/generate", json={"prompt": "hi", "max_new_tokens": 2})
        assert fake.entered.wait(timeout=5)
        assert c.get("/healthz").status_code == 200
        fake.proceed.set()
        assert pending.result(timeout=5).status_code == 200
