"""FastAPI application that serves one Backend over HTTP.

Start it with::

    uvicorn ladder.serve.app:create_app --factory --host 0.0.0.0 --port 8000

There is deliberately no module-level ``app = create_app()``: building the app
from environment variables loads a model, and that should happen when the server
starts, not whenever this module is imported (for example by the tests).

Nothing here imports torch, transformers or onnxruntime at import time. The real
backend is built lazily in the startup hook through ``ladder.registry``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from ladder.backends.base import Backend
from ladder.serve.schemas import GenerateRequest, GenerateResponse

DEFAULT_RUNG = "01_torch_fp32"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_backend_from_env() -> Backend:
    """Build the backend named by the ``LADDER_*`` environment variables.

    Reading the rung from the environment lets one container image serve any
    rung. The registry is imported here, not at the top of the module, because
    it pulls in the heavy model libraries.
    """
    from ladder.registry import build_backend

    rung = os.environ.get("LADDER_RUNG", DEFAULT_RUNG)
    model_id = os.environ.get("LADDER_MODEL", DEFAULT_MODEL)
    threads_raw = os.environ.get("LADDER_THREADS", "").strip()
    try:
        num_threads = int(threads_raw) if threads_raw else None
    except ValueError as err:
        raise ValueError(f"LADDER_THREADS must be an integer, got {threads_raw!r}") from err
    return build_backend(rung=rung, model_id=model_id, num_threads=num_threads)


def _run_generation(backend: Backend, prompt: str, max_new_tokens: int) -> GenerateResponse:
    """Tokenize, generate and detokenize one request. Runs on the worker thread.

    Tokenizing and decoding happen here too, not on the event loop, so that every
    call into the backend goes through the same single thread.
    """
    prompt_ids = backend.encode(prompt)
    result = backend.generate(prompt_ids, max_new_tokens)
    return GenerateResponse(
        text=backend.decode(result.token_ids),
        prompt_tokens=result.prompt_tokens,
        generated_tokens=len(result.token_ids),
        ttft_ms=result.ttft_s * 1000.0,
        total_ms=result.total_s * 1000.0,
    )


def create_app(backend: Backend | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        backend: An already loaded backend (the tests pass a fake one). If None,
            the backend is built from environment variables at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Only a backend we built ourselves is ours to close.
        owns_backend = backend is None
        loaded = load_backend_from_env() if owns_backend else backend
        # Concurrency model (M0): inference is CPU-bound and backends are not
        # thread-safe, so every generation runs on ONE dedicated worker thread.
        # Requests queue up inside the executor and are served strictly one at a
        # time, while the event loop stays free to answer /healthz and /info.
        # Milestone M5 replaces this with dynamic batching.
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-gen")
        app.state.backend = loaded
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=True)
            app.state.backend = None
            if owns_backend:
                loaded.close()

    app = FastAPI(title="inference-ladder", lifespan=lifespan)
    # None until startup has finished; the endpoints answer 503 until then.
    app.state.backend = None

    def get_backend(request: Request) -> Backend:
        """Return the loaded backend, or fail with 503 if it is not ready yet."""
        loaded = request.app.state.backend
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return loaded

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, str]:
        """Liveness: 200 once the model is loaded, 503 before."""
        get_backend(request)
        return {"status": "ok"}

    @app.get("/info")
    def info(request: Request) -> dict[str, Any]:
        """The loaded backend's BackendInfo, field for field."""
        return dataclasses.asdict(get_backend(request).info)

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        """Generate a continuation of the prompt on the single worker thread."""
        loaded = get_backend(request)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            request.app.state.executor,
            _run_generation,
            loaded,
            body.prompt,
            body.max_new_tokens,
        )

    return app
