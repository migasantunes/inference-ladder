# Service, container and load test

Status: skeleton in M0, completed in milestone M5.

## What it is

A small HTTP service that wraps one `Backend`, packaged as a container image,
with a load test that produces the requests-per-second figure.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness. Returns 200 once the model is loaded. |
| `GET /info` | The loaded backend's `BackendInfo` (rung, engine, precision, threads). |
| `POST /generate` | Body: `{"prompt": str, "max_new_tokens": int}`. Returns the generated text, token counts, and timing (`ttft_ms`, `total_ms`). |

The rung to load comes from environment variables (`LADDER_RUNG`,
`LADDER_MODEL`, `LADDER_THREADS`), so one image can serve any rung.

## Concurrency model

Model inference is CPU-bound and the backends are not safe to call from several
threads at once. M0 uses the simplest correct design: a single worker runs
generations one at a time; requests wait in a queue.

## Dynamic batching (M5)

Theory: one forward pass over a batch of B sequences costs much less than B
separate passes, because the matrix multiplications share the same weights and
the hardware vectorises across the batch. A batcher collects requests that
arrive within a short window (a few milliseconds) and runs them together.

- Throughput rises with batch size.
- Each request may wait up to the window before starting, and a batch finishes
  when its longest member finishes, so individual latency can rise.
- The window and the maximum batch size are the two knobs. The load test sweeps
  them.

Batched generation needs padding and an attention mask, and sequences of
different lengths in one batch. This requires extending the backend interface
with a batched generate; design that in M5, not before.

## Container

- Slim Python base image, non-root user, dependencies installed from
  `pyproject.toml` extras, model either baked in at build time or mounted.
- CPU-only PyTorch wheels to keep the image small.
- Run with explicit limits so the numbers mean something:
  `docker run --cpus=2 --memory=2g -p 8000:8000 ...`

## Load test

Locust (or equivalent) with a fixed prompt mix and fixed `max_new_tokens`.
Report requests/s and p50/p95/p99 latency at several concurrency levels, with
and without batching, at the stated CPU limit. Results go in
`results/<machine>/load/`.

## Acceptance (M5)

From a clean clone: build the image, run it, hit `/generate`, run the load test,
and get the table in the README.
