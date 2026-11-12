"""Build ``results/<machine>/table.md`` from the result files of one machine.

One row per rung. Ratios are against the baseline rung (01_torch_fp32) and are
written so that bigger is better: decode speedup = rung / baseline tokens/s,
size reduction = baseline size / rung size. Without a baseline result the ratio
columns are left out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ladder.bench.results import read_machine_results
from ladder.registry import BASELINE_RUNG

MISSING = "-"


def _fmt(value: Any, spec: str) -> str:
    """Format a number, or return "-" when it is missing."""
    return MISSING if value is None else format(value, spec)


def _ratio(numerator: float | None, denominator: float | None) -> str:
    if not numerator or not denominator:  # missing, or 0 meaning "unknown"
        return MISSING
    return f"{numerator / denominator:.2f}x"


def _metrics(runs: dict[str, dict], kind: str) -> dict[str, Any]:
    return runs.get(kind, {}).get("metrics", {})


def _size_mb(runs: dict[str, dict]) -> float | None:
    """Size on disk from either result kind; 0 means unknown, shown as missing."""
    for kind in ("bench", "quality"):
        size = runs.get(kind, {}).get("backend", {}).get("size_on_disk_bytes")
        if size:
            return size / 2**20
    return None


def _backend_field(runs: dict[str, dict], field: str) -> str:
    for kind in ("bench", "quality"):
        value = runs.get(kind, {}).get("backend", {}).get(field)
        if value:
            return str(value)
    return MISSING


def _tokens_match(runs: dict[str, dict], baseline: dict[str, dict] | None, rung: str) -> str:
    """ "yes" if every prompt's token hash equals the baseline's."""
    if rung == BASELINE_RUNG:
        return "(baseline)"
    ours = _metrics(runs, "bench").get("token_hashes")
    theirs = _metrics(baseline or {}, "bench").get("token_hashes")
    if not ours or not theirs:
        return MISSING
    return "yes" if ours == theirs else "no"


def render_table(results: dict[str, dict[str, dict]]) -> str:
    """Markdown table for ``{rung: {kind: record}}``, as read from disk."""
    baseline = results.get(BASELINE_RUNG)
    with_ratios = baseline is not None
    base_tps = _metrics(baseline or {}, "bench").get("decode_tps_median")
    base_size = _size_mb(baseline) if baseline else None

    header = ["rung", "engine", "precision", "decode tok/s"]
    header += ["decode speedup"] if with_ratios else []
    header += ["e2e tok/s", "TTFT p50 ms", "TTFT p95 ms", "ITL p50 ms", "ITL p99 ms"]
    header += ["RSS after load MB", "peak RSS MB", "size MB"]
    header += ["size reduction"] if with_ratios else []
    header += ["perplexity", "top-1 agreement", "mean KL (nats)", "tokens = baseline"]

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for rung in sorted(results):
        runs = results[rung]
        bench = _metrics(runs, "bench")
        quality = _metrics(runs, "quality")
        size = _size_mb(runs)

        row = [rung, _backend_field(runs, "engine"), _backend_field(runs, "precision")]
        row += [_fmt(bench.get("decode_tps_median"), ".1f")]
        row += [_ratio(bench.get("decode_tps_median"), base_tps)] if with_ratios else []
        row += [
            _fmt(bench.get("e2e_tps_median"), ".1f"),
            _fmt(bench.get("ttft_ms_p50"), ".2f"),
            _fmt(bench.get("ttft_ms_p95"), ".2f"),
            _fmt(bench.get("itl_ms_p50"), ".2f"),
            _fmt(bench.get("itl_ms_p99"), ".2f"),
            _fmt(bench.get("rss_after_load_mb"), ".1f"),
            _fmt(bench.get("peak_rss_mb"), ".1f"),
            _fmt(size, ".1f"),
        ]
        row += [_ratio(base_size, size)] if with_ratios else []
        row += [
            _fmt(quality.get("perplexity"), ".3f"),
            _fmt(quality.get("top1_agreement"), ".4%"),
            _fmt(quality.get("mean_kl"), ".3g"),
            _tokens_match(runs, baseline, rung),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_report(out_dir: str | Path, machine: str) -> Path:
    """Read every result for ``machine`` and write its ``table.md``."""
    results = read_machine_results(out_dir, machine)
    if not results:
        raise FileNotFoundError(f"No result files in {Path(out_dir) / machine}")
    path = Path(out_dir) / machine / "table.md"
    if BASELINE_RUNG in results:
        note = f"Ratios are against {BASELINE_RUNG}; higher is better."
    else:
        note = f"No {BASELINE_RUNG} result, so ratio columns are omitted."
    note += ' "-" means not measured.'
    path.write_text(f"# Results: {machine}\n\n{render_table(results)}\n{note}\n", encoding="utf-8")
    return path
