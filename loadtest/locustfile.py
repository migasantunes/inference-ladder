"""Locust load test for the inference-ladder service.

Start the container first (see loadtest/README.md), then run, from the repo root:

    pip install -e ".[load]"
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless --users 4 --spawn-rate 4 --run-time 2m --csv results/<machine>/load/u4

Locust is an optional dependency. Nothing in the package or the tests imports
this file.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# A fixed, short prompt mix so that runs are comparable with each other.
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "The main advantage of quantizing a neural network is",
    "def fibonacci(n):",
    "A good way to start the morning is",
    "The three laws of thermodynamics state that",
    "In machine learning, overfitting happens when",
    "The recipe for a simple tomato soup begins with",
]

# Fixed so that every request costs the same amount of decoding work.
MAX_NEW_TOKENS = 32


class GenerateUser(HttpUser):
    """One simulated client that sends generation requests in a loop."""

    # A short pause between requests from the same user.
    wait_time = between(0.1, 0.5)

    @task
    def generate(self) -> None:
        """POST one prompt to /generate."""
        self.client.post(
            "/generate",
            json={"prompt": random.choice(PROMPTS), "max_new_tokens": MAX_NEW_TOKENS},
            name="/generate",
        )
