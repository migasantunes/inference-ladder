# Rungs 03 and 04 - int8 quantization

Status: not started (milestone M3).

## What it is

Weights (and for rung 04, activations) stored and multiplied as 8-bit integers
instead of 32-bit floats.

## Theory

A float tensor `x` is approximated by an integer tensor `q` and two constants:

    x ≈ scale × (q − zero_point)

`scale` and `zero_point` are chosen so the tensor's value range maps onto the 256
values an int8 can hold. Consequences:

- **Size:** one byte per weight instead of four.
- **Speed:** less memory traffic, and integer vector instructions on modern CPUs.
- **Error:** every value is rounded to one of 256 levels. A tensor with a wide
  range or outliers gets coarse steps, and the rounding error propagates through
  the network.

Two ways to pick the ranges:

- **Dynamic (rung 03):** weights are quantized ahead of time. Activation ranges
  are computed on the fly for each input. No calibration data needed; some
  runtime overhead.
- **Static (rung 04):** activation ranges are fixed in advance by running sample
  inputs (calibration) and recording what was observed. Less overhead at run
  time, but wrong ranges on unseen inputs hurt quality. Transformers are known to
  be sensitive here, so a poor result is plausible and worth reporting.

Per-channel quantization (a separate scale per output channel of a weight
matrix) usually reduces error; treat it as an option to measure.

## Implementation notes

- ONNX Runtime ships a quantization toolkit with dynamic and static entry
  points. Check the current documentation for supported operators, the
  recommended pre-processing step, and the CPU instruction-set guidance, since
  the best settings depend on the processor.
- Calibration data: a few hundred windows from WikiText-2 **train**.
- Each variant is a separate artifact and a separate rung name; `OrtBackend`
  from rung 02 loads them unchanged.
- Record in the result: quantization mode, per-channel on or off, operators
  quantized, calibration method and sample count.

## Acceptance

Both rungs in the table with size, speed, perplexity, top-1 agreement and KL
against rung 01. No quality threshold: whatever happens is the result.

## Questions the defend note must answer

- What are scale and zero-point, and how are they chosen?
- Why is the model about 4× smaller but not 4× faster?
- Dynamic versus static: what is traded?
- Which operators were left in float, and why?
