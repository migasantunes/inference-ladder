# Rung 02 - ONNX Runtime, float32

Status: not started (milestone M2).

## What it is

The same weights, exported from PyTorch to an ONNX graph and executed by ONNX
Runtime on the CPU execution provider, with graph optimizations enabled.

## Theory

PyTorch eager mode runs the model as Python code. ONNX is a file format that
describes the model as a fixed graph of operations. Because the whole graph is
known before anything runs, the runtime can rewrite it once, ahead of time:

- **Constant folding:** anything computable from the weights alone is computed at
  load time.
- **Operator fusion:** common sequences (for example matmul + add, or the pieces
  of layer normalisation or GELU) are replaced by one optimized kernel.
- **No Python in the loop:** execution stays inside native code.

The numbers should match the baseline up to floating-point rounding, because the
arithmetic is the same.

## Implementation notes

- Export with the KV-cache: the exported graph must take past keys and values as
  inputs and return updated ones. Hugging Face Optimum has an exporter for
  causal language models that does this; check its current documentation for the
  exact command and options before writing code.
- `OrtBackend` implements the same greedy loop as the PyTorch backend, feeding
  cache tensors back in each step. Prefer driving `onnxruntime.InferenceSession`
  directly over a high-level wrapper, so timing and cache handling stay visible.
  Look at I/O binding to avoid copying the cache tensors every step.
- Session options to record in the result: graph optimization level, intra-op
  thread count.
- Exported files go in `artifacts/` (git-ignored). The export command is a CLI
  subcommand so it is reproducible.

## Acceptance

- `top1_agreement ≥ 0.999` against rung 01 and identical token hashes on the
  prompt set. If not, the export is wrong; fix that before measuring speed.
- Bench and quality results committed; defend note written.

## Questions the defend note must answer

- What is in an `.onnx` file?
- Why can a static graph be optimized in ways eager PyTorch cannot?
- Where did the speedup (or lack of it) come from on this model?
