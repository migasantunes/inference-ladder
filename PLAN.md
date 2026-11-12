# Project plan

## Goal

Take one small open language model and make it progressively cheaper to run, one
technique per rung, measuring every rung with the same protocol. Then serve the
best variant behind an HTTP endpoint in a container and load-test it.

The target outcome is one honest sentence backed by a reproducible table:

> Reduced latency N× and model size M× at under X points of quality loss, served
> at R requests/s on 2 vCPUs.

N, M, X and R are unknown until the runs happen. Nothing in this repo may state a
performance number that did not come out of `results/`.

## Non-goals

- Beating llama.cpp, vLLM or TensorRT. They are the reference point, not the
  competition (see milestone M7).
- Training a new model. The only training in scope is the optional distillation
  stretch goal.
- A product. The service exists to prove the optimized model is deployable.

## Fixed choices

| Choice | Value | Why |
|---|---|---|
| Primary model | `HuggingFaceTB/SmolLM2-135M` | Small enough to iterate on a laptop CPU, permissive licence, real architecture (not a toy) |
| Second model | `HuggingFaceTB/SmolLM2-360M` | Checks that conclusions hold at another size |
| Primary hardware target | CPU, thread count pinned | Matches the "cheap box" deployment story; int8 is a CPU technique |
| Secondary hardware | The owner's CUDA GPU, PyTorch rungs only | Extra column, not the headline |
| Decoding | Greedy, fixed `max_new_tokens`, no early stop | Deterministic, so rungs differ only by engine |
| Quality data | WikiText-2 (raw) test split | Standard, small, public |
| Language | Python ≥ 3.10 | |

If SmolLM2 turns out to export badly, substitute another causal LM under 500M
parameters and record the change here. Everything else in the plan is
model-agnostic.

## The ladder

Each rung is one `Backend` implementation (`src/ladder/backends/`) plus a spec in
`docs/rungs/`. Rungs are compared against `01_torch_fp32`.

| Rung | What changes | Expected effect | Spec |
|---|---|---|---|
| `00_torch_nocache` | Baseline with the KV-cache switched off | Much slower; shows what the cache buys | [00](docs/rungs/00_torch_nocache.md) |
| `01_torch_fp32` | **Baseline.** PyTorch, float32, KV-cache on | Reference | [01](docs/rungs/01_torch_fp32.md) |
| `02_ort_fp32` | Export to ONNX, run in ONNX Runtime with graph optimizations | Faster, same quality to rounding | [02](docs/rungs/02_ort_fp32.md) |
| `03_ort_int8_dynamic` | Dynamic int8 quantization of weights | ~4× smaller, faster, small quality loss | [03](docs/rungs/03_ort_int8.md) |
| `04_ort_int8_static` | Static int8 with calibration data | Possibly faster again, higher quality risk | [03](docs/rungs/03_ort_int8.md) |
| `05_ort_int8_mixed` | Sensitivity study: quantize all but the layers that hurt most | Recovers quality at small size cost | [05](docs/rungs/05_sensitivity.md) |
| `ref_llamacpp` | Same model as GGUF in llama.cpp | External yardstick | [ref](docs/rungs/ref_llamacpp.md) |

"Expected effect" is a hypothesis. A rung that fails to deliver (static
quantization of transformers often does) stays in the table with its real
numbers and an explanation; negative results are part of the write-up.

## Milestones

Work one milestone per branch and pull request. A milestone is done only when
its acceptance criteria hold.

### M0 - Scaffold and harness (done in the bootstrap session)

- Package layout, `Backend` interface, benchmark harness, PyTorch backend, CLI.
- Unit tests run on a tiny randomly initialised model, with no downloads.
- FastAPI service skeleton, Dockerfile, CI running lint and tests.

Not verified in M0: anything involving the real model. The bootstrap environment
could not reach Hugging Face.

### M1 - Baseline numbers on real hardware

- `ladder bench` and `ladder quality` run for `00_torch_nocache` and
  `01_torch_fp32` on SmolLM2-135M on the owner's machine.
- Results committed under `results/<machine>/`.
- Sanity check: baseline perplexity is in a plausible range for a model this
  size, and `00` is clearly slower than `01` with identical output tokens.
- **Accept:** `ladder report` renders a two-row table; `docs/defend/01` updated
  with the real numbers.

### M2 - ONNX Runtime

- Export with the KV-cache preserved; `OrtBackend` implemented.
- **Accept:** top-1 agreement with baseline ≥ 99.9% and identical greedy output
  on the prompt set; speed and memory recorded; defend note written.

### M3 - Int8 quantization

- Dynamic, then static (calibration on WikiText-2 *train*, never test).
- **Accept:** both rungs in the table with size, speed, perplexity delta,
  top-1 agreement and KL; defend note explains scale/zero-point and why static
  differs from dynamic.

### M4 - Sensitivity study

- Quantize one layer group at a time, measure the quality hit, rank the groups.
- Build `05_ort_int8_mixed` by excluding the worst offenders.
- **Accept:** a chart of quality loss per layer group, and a mixed rung that
  sits between fp32 and full int8 on the size/quality curve. This is the
  project's own finding; give it space in the README.

### M5 - Service, container, load test

- Service loads the best rung; dynamic batching with a tunable wait window.
- Dockerfile builds a runnable image with the model baked in or mounted.
- Locust run at fixed CPU limits (`docker run --cpus=2`).
- **Accept:** requests/s and latency percentiles at a stated concurrency,
  with and without batching; `docker run` instructions that work from a clean
  clone.

### M6 - Reference and regression gate

- `ref_llamacpp` row (Q8_0 and Q4_K_M) measured with the same prompts.
- CI job that benchmarks the previous and the current commit **in the same
  job** on the tiny model and fails on a relative slowdown beyond a threshold.
  Shared runners are too noisy for absolute thresholds.
- **Accept:** reference rows in the table; a deliberately slowed commit fails CI.

### M7 - Write-up

- README with the results table, the speed-versus-quality chart, the
  sensitivity chart, and a "what I would do next" section.
- One CV bullet and one short LinkedIn post drafted from the real numbers.

### Stretch

- Repeat the ladder on SmolLM2-360M.
- GPU column: fp16 and `torch.compile` for the PyTorch rungs.
- Distillation of a smaller student from the 360M model.
- Deploy the container to a small cloud instance.

## How the work is organised

- The lead model plans, writes specs, reviews and decides. Implementation is
  delegated to sub-agents, one per rung or component, each working against the
  spec in `docs/rungs/` and the interface in `src/ladder/backends/base.py`.
- Every rung ships as: backend code, tests on the tiny model, a results file from
  a real run, an updated spec, and a defend note.
- The owner runs every benchmark that is reported, on his own machine.

## Risks

| Risk | Mitigation |
|---|---|
| Export of the model with KV-cache fails or is awkward | Fall back to another small causal LM; the harness does not care |
| Static quantization destroys quality | Report it; the sensitivity study is the answer to it |
| Timing noise on a laptop | Pinned threads, warmup, repeats, median and percentiles, machine on mains power with nothing else running |
| Library APIs differ from what the specs assume | Specs describe intent; check current documentation before coding each rung |
| The owner cannot explain a rung in an interview | Defend notes are a required deliverable, and each rung is small |
