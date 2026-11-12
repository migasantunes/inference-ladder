# Rung 01 - PyTorch float32 (baseline)

Status: implemented in M0, measured on the real model in M1.

## What it is

The model as published: 32-bit float weights, run with PyTorch in eager mode on
CPU, KV-cache on. Every other rung is reported as a ratio against this one.

## Implementation

`TorchBackend(use_cache=True)` in `src/ladder/backends/torch_backend.py`.

- Loads with `AutoModelForCausalLM` and `AutoTokenizer`, `eval()` mode,
  `torch.inference_mode()` around every forward pass.
- Generation is a hand-written greedy loop, not `model.generate`. First step:
  the whole prompt goes in and returns logits plus `past_key_values`. Each later
  step: only the last token and the cache go in. A timestamp is taken after each
  argmax. Writing the loop by hand keeps the per-token timing exact and makes the
  cache explicit.
- `torch.set_num_threads` is set from the `--threads` option.
- `size_on_disk_bytes` is the size of the weight files in the model directory.

## What limits its speed

- Float32 matrix multiplications dominate; on CPU they are bound by memory
  bandwidth as much as by arithmetic.
- Eager mode dispatches each operation from Python, one at a time, with no
  knowledge of what comes next, so nothing is fused or precomputed.

## Acceptance

Benchmarks and quality run end to end on SmolLM2-135M; results committed.
