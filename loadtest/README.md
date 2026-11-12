# Load test

1. Build and start the service with explicit CPU and memory limits, so the numbers mean something:

       docker build -t inference-ladder .
       docker run --rm --cpus=2 --memory=2g -p 8000:8000 \
           -e LADDER_THREADS=2 -v ladder-models:/models inference-ladder

2. Wait until `curl -f http://localhost:8000/healthz` returns `{"status":"ok"}` (the first start downloads the model).
3. Install Locust (`pip install -e ".[load]"`) and run the exact command in the docstring of `locustfile.py`, changing `--users` to set the concurrency.
4. Locust's `--csv` output gives requests/s and p50/p95/p99 latency. Save it under `results/<machine>/load/`, and record the `--cpus` limit and the rung (`GET /info`) next to it.

The M0 service runs one generation at a time, so this measures the unbatched baseline only.
Dynamic batching, and the sweeps over batch window and maximum batch size, come in milestone M5.
