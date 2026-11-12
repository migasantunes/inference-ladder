# Benchmark protocol

Every rung is measured the same way. If a number in this repo was not produced by
this protocol, it does not go in the results table.

## Isolation

- One rung per process. `ladder bench --rung X` loads exactly one backend, so
  memory figures are not polluted by another model.
- Thread count is pinned (`--threads`, default: physical cores) and recorded.
- Runs happen on mains power with other heavy programs closed. The harness
  cannot enforce this; it records the machine fingerprint so runs from different
  machines are never mixed.

## Speed (`ladder bench`)

Inputs: the fixed prompt set in `data/prompts.jsonl` (16 prompts of varied
length). Decoding is greedy with `max_new_tokens = 64` and no early stop.

Procedure:

1. Load the backend; record load time and resident memory after load.
2. Warm up: 3 generations, discarded.
3. Measure: `repeats = 5` passes over the prompt set (80 generations).

Reported per rung:

| Metric | Definition |
|---|---|
| `ttft_ms_p50`, `ttft_ms_p95` | Time to first token: call start until the first new token is available. Includes processing the whole prompt. |
| `itl_ms_p50`, `itl_ms_p95`, `itl_ms_p99` | Inter-token latency: gap between consecutive tokens, pooled over all generations, first token excluded. |
| `decode_tps_median` | Per generation, `(tokens − 1) / (total − ttft)`; median over generations. Steady-state decoding speed. |
| `e2e_tps_median` | Per generation, `tokens / total`; median over generations. What a caller experiences. |
| `load_time_s` | Wall time to construct the backend. |
| `rss_after_load_mb`, `peak_rss_mb` | Process resident memory, sampled in a background thread during the run. |
| `size_on_disk_mb` | Total size of the weight files the backend loads. |

Medians and percentiles are used instead of means because timing noise is
one-sided: a run can only be slowed down by interference, never sped up.

The harness also stores a hash of the generated token ids per prompt. Two rungs
with the same hash produced identical text.

## Quality (`ladder quality`)

Data: WikiText-2 raw, test split, joined with blank lines and tokenized once.
Split into non-overlapping windows of 512 tokens.

| Metric | Definition |
|---|---|
| `perplexity` | `exp` of the mean negative log-likelihood of each token given the previous tokens in its window. Lower is better. |
| `top1_agreement` | Fraction of positions where the rung's most likely next token equals the baseline's. Needs `--reference`. |
| `mean_kl` | Mean KL divergence from the baseline's next-token distribution to the rung's, in nats per position. Needs `--reference`. |

Perplexity says how good the model is in absolute terms. Agreement and KL say
how far the optimized model drifted from the original, which is the more
sensitive measure: perplexity can stay flat while individual predictions change.

`--reference 01_torch_fp32` loads the baseline in the same process and compares
logits window by window. Quality runs are not used for memory figures.

`--max-windows` limits the evaluation for quick iteration. Reported results use
the full test split; the number of windows is stored in the result.

## Calibration data

Static quantization and the sensitivity study need sample inputs. These come
from the WikiText-2 **train** split only. Test data is never used to tune a rung.

## Output

Each run writes one JSON file to `results/<machine>/<rung>.<kind>.json`
(`kind` is `bench` or `quality`) containing the metrics, the settings, and the
environment: CPU model, core count, RAM, OS, Python and library versions, thread
count, git commit, timestamp.

`<machine>` is a short name passed with `--machine` (for example `laptop-cpu`).
`ladder report --machine <machine>` reads the directory and writes
`results/<machine>/table.md` with one row per rung and ratios against the
baseline.

## What the unit tests cover

Tests build a tiny randomly initialised GPT-2 (two layers, small vocabulary) in
memory. They check that the harness, the metrics and the backends are wired
correctly. They say nothing about the real model's speed or quality.
