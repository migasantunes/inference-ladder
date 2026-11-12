# Defend it: measurement basics and the KV-cache

Notes for explaining rungs 00 and 01 out loud. Numbers get added after the M1
runs; until then this covers the reasoning only.

## How a language model generates text

The model takes a sequence of token ids and outputs, for each position, a score
(logit) for every token in the vocabulary. The scores at the last position
predict the next token. Generation is a loop: run the model, pick a token,
append it, repeat. Picking the highest-scoring token every time is greedy
decoding.

So the cost of generating T tokens is T forward passes, and each forward pass is
mostly matrix multiplications inside the attention and MLP layers.

## Why two speed numbers

- **Time to first token (TTFT):** the first step must process the entire prompt.
  It scales with prompt length.
- **Inter-token latency / tokens per second:** later steps process one token
  each (with the cache). It is roughly constant per token.

A chat user feels TTFT as the pause before the answer starts and tokens per
second as the typing speed. Averaging them into one number hides both.

## Why percentiles and medians

Timing noise is one-sided. Background processes, CPU frequency changes and
cache misses only ever make a run slower. A mean is dragged up by a few bad
runs; the median describes the typical case and p95/p99 describe the bad cases.
Services are judged on tail latency because that is what the unluckiest users
see.

## Why warm up

The first few runs pay one-off costs: memory allocation, lazy initialisation
inside the libraries, cold CPU caches. They are not representative, so they are
run and thrown away.

## Perplexity

For each token in a test text, the model assigned some probability p to the
token that actually came next. Perplexity is `exp(mean(−ln p))`. Intuition: if
perplexity is 20, the model is on average as uncertain as if it were choosing
uniformly among 20 tokens. Lower is better. It measures the model against real
text.

## Agreement and KL divergence

These measure an optimized model against the original model rather than against
text. Top-1 agreement is how often both pick the same next token. KL divergence
measures how different the two full probability distributions are; it is zero
only when they are identical. They detect drift that perplexity can miss,
because perplexity can stay level while predictions change in both directions.

## The KV-cache

Inside attention, each token produces a query, a key and a value. A token's
query is matched against the keys of itself and all earlier tokens, and the
result is a weighted mix of their values.

In a causal model, a token only attends backwards. Appending a new token
therefore does not change the keys and values of the tokens before it. The
KV-cache stores them per layer. Each generation step then:

1. runs only the new token through the model,
2. computes its query, key and value in each layer,
3. attends over the cached keys and values plus its own,
4. appends its key and value to the cache.

Without the cache, step n recomputes everything for all n tokens. Total work for
T tokens goes from order T² to order T token-passes.

The price is memory: the cache holds two tensors per layer for every token so
far, so it grows linearly with sequence length.

## Likely interview questions

- *Why does TTFT grow with prompt length but tokens per second barely does?*
  The first step processes every prompt token; later steps process one.
- *Does the KV-cache change the output?* No. It stores intermediate results that
  would be recomputed identically. Rung 00 and rung 01 produce the same tokens.
- *Does attention still cost more for longer sequences with the cache?* Yes: the
  new token still attends over all cached positions, so per-token cost grows
  slowly with length. The cache removes the recomputation, not the attention.
- *Why greedy decoding in the benchmark?* It is deterministic, so two engines
  can be checked token for token.
- *Why not use the mean latency?* See percentiles above.
- *What would you measure differently for a real service?* Tail latency under
  concurrent load, which is what the load test in milestone M5 does.
