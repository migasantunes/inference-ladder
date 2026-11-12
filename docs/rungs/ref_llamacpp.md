# Reference - llama.cpp

Status: not started (milestone M6).

## Purpose

An external yardstick. llama.cpp is a mature, heavily optimized CPU inference
engine. Running the same model through it shows how far this project's best rung
is from what a specialised engine achieves, and stops the write-up from
overclaiming.

## Method

- Convert the model to GGUF (or use an existing conversion if one is published)
  at two precisions: 8-bit and a 4-bit variant.
- Run the same prompt set, greedy decoding, same `max_new_tokens`, same thread
  count, and record TTFT, decode tokens/s, memory and file size.
- Implement as a `Backend` if the Python bindings expose per-token timing;
  otherwise as a separate script that writes a result file in the same format.

## Caveats to state in the write-up

- Quantization schemes differ from ONNX Runtime's, so this is a comparison of
  end results, not of like-for-like techniques.
- Tokenization and prompt handling must match; verify by comparing generated
  text on a few prompts.

## Acceptance

Reference rows appear in the results table, marked as reference.
