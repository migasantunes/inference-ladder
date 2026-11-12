"""Request and response bodies for the HTTP API.

Pydantic checks the request before our code runs, so an empty prompt or an
out-of-range ``max_new_tokens`` is rejected with a 422 and never reaches the model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Upper bound on generated tokens per request. It keeps one request from holding
# the single worker for a very long time while everyone else waits in the queue.
MAX_NEW_TOKENS_LIMIT = 256


class GenerateRequest(BaseModel):
    """Body of ``POST /generate``."""

    prompt: str = Field(min_length=1, description="Text to continue. Must not be empty.")
    max_new_tokens: int = Field(
        default=64,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
        description="Exactly this many tokens are generated (greedy, no early stop).",
    )


class GenerateResponse(BaseModel):
    """Body returned by ``POST /generate``.

    The timings come from the backend and cover the generation only. Time spent
    waiting in the queue for the worker is not included; the load test measures
    end-to-end latency from the client side.
    """

    text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    total_ms: float
