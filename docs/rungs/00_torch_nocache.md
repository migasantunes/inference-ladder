# Rung 00 - PyTorch without the KV-cache

Status: implemented in M0, measured on the real model in M1.

## What it is

The baseline model with `use_cache=False`. For every new token the whole
sequence so far is fed through the model again.

This is an ablation, not a step forward. It exists to show how much the KV-cache
(on by default everywhere) is already doing, and to make the idea concrete
before the later rungs have to preserve it through an export.

## Why it is slow

In each attention layer, a token's query is compared with the keys of all
earlier tokens, and the matching values are mixed in. Keys and values of earlier
tokens do not change when a new token is appended, because attention in a causal
model only looks backwards.

Without a cache, generating token `n` recomputes keys and values for all `n − 1`
earlier tokens in every layer. Generating `T` tokens therefore costs on the order
of `T²` token-passes. With the cache, each step processes one new token and
reuses the stored keys and values: on the order of `T` token-passes.

## Implementation

`TorchBackend(use_cache=False)`: the greedy loop passes the full `input_ids`
each step and ignores `past_key_values`.

## Expected result

- Identical generated tokens to rung 01 (same hash in the bench output).
- Inter-token latency that grows with sequence position, where rung 01 stays
  roughly flat.
- Identical quality metrics: the cache changes cost, not results.

## Acceptance

Token hashes match rung 01 on every prompt, and `decode_tps_median` is clearly
lower.
