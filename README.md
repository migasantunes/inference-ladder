# inference-ladder

An inference-optimization ladder for a small language model. One technique per
rung, every rung measured with the same protocol, and the best variant served
behind an HTTP endpoint in a container.

**Status: scaffold.** The benchmark harness, the PyTorch baseline and the
service skeleton are implemented and unit-tested on a tiny random model. No rung
has been measured on the real model yet, so there are no results to show. The
results table and charts will appear here once the runs in milestone M1 onwards
are done.

## The ladder

| Rung | What changes | State |
|---|---|---|
| `00_torch_nocache` | Baseline with the KV-cache off (ablation) | Implemented, not measured |
| `01_torch_fp32` | Baseline: PyTorch, float32 | Implemented, not measured |
| `02_ort_fp32` | ONNX Runtime with graph optimizations | Planned |
| `03_ort_int8_dynamic` | Dynamic int8 quantization | Planned |
| `04_ort_int8_static` | Static int8 quantization with calibration | Planned |
| `05_ort_int8_mixed` | Mixed precision from a per-layer sensitivity study | Planned |
| `ref_llamacpp` | llama.cpp as an external reference | Planned |

Model: [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

## Where things are

| Path | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | Milestones, fixed choices, acceptance criteria |
| [`docs/benchmark-protocol.md`](docs/benchmark-protocol.md) | Exactly what is measured and how |
| [`docs/rungs/`](docs/rungs) | One spec per rung |
| [`docs/defend/`](docs/defend) | Plain-language notes on each idea |
| `src/ladder/backends/` | One backend per way of running the model |
| `src/ladder/bench/` | Speed, memory and quality measurement |
| `src/ladder/serve/` | FastAPI service |
| `results/<machine>/` | Result files and the generated table (once runs exist) |

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,serve,data]"
    pytest                       # unit tests, no downloads

Measure the baseline (downloads the model on first use):

    ladder bench   --rung 01_torch_fp32    --machine my-laptop
    ladder bench   --rung 00_torch_nocache --machine my-laptop
    ladder quality --rung 01_torch_fp32    --machine my-laptop
    ladder report  --machine my-laptop     # writes results/my-laptop/table.md

Run the service:

    uvicorn ladder.serve.app:create_app --factory --port 8000
    curl -s localhost:8000/generate -H 'content-type: application/json' \
         -d '{"prompt": "The capital of Portugal is", "max_new_tokens": 16}'

## Measurement in one paragraph

Decoding is greedy with a fixed number of new tokens, so every rung produces a
checkable sequence. Speed is reported as time to first token and inter-token
latency (median and tail percentiles) over a fixed prompt set after warmup.
Quality is perplexity on WikiText-2 plus agreement and KL divergence against the
float32 baseline. Each rung runs in its own process with a pinned thread count,
and every result file records the machine it came from.
