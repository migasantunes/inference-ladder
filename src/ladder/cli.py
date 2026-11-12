"""Command line entry point: ``ladder bench | quality | report | rungs``.

Heavy libraries (torch, transformers, datasets) are only imported when a
backend is built or a dataset is loaded, so ``ladder rungs`` and ``--help`` are
instant.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import psutil

from ladder.bench.env import environment
from ladder.bench.memory import RssSampler, current_rss_bytes
from ladder.bench.quality import evaluate_quality, load_text_file, load_wikitext2, make_windows
from ladder.bench.results import write_result
from ladder.bench.speed import load_prompts, run_speed, speed_metrics, token_hashes
from ladder.registry import BASELINE_RUNG, available_rungs, build_backend
from ladder.report import write_report

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
MB = 2**20


def fmt(value: object) -> str:
    """Short display form of a metric for the console summary."""
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def default_threads() -> int:
    """Physical core count, the protocol's default thread setting."""
    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1


def cmd_bench(args: argparse.Namespace) -> None:
    threads = args.threads or default_threads()
    prompts = load_prompts(args.prompts)

    # The sampler starts before loading, so the peak covers load and run.
    with RssSampler() as mem:
        backend = build_backend(args.rung, args.model, threads)
        rss_after_load = current_rss_bytes()
        measurements = run_speed(backend, prompts, args.max_new_tokens, args.warmup, args.repeats)

    info = backend.info
    hashes, consistent = token_hashes(measurements)
    metrics = speed_metrics(measurements)
    metrics.update(
        load_time_s=info.load_time_s,
        rss_after_load_mb=rss_after_load / MB,
        peak_rss_mb=mem.peak_rss_bytes / MB,
        size_on_disk_mb=info.size_on_disk_bytes / MB,
        token_hashes=hashes,
        tokens_consistent_across_repeats=consistent,
    )
    settings = {
        "model": args.model,
        "threads": threads,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "prompts_file": str(args.prompts),
        "num_prompts": len(prompts),
        "num_generations": len(measurements),
    }
    path = write_result(
        args.out_dir,
        args.machine,
        args.rung,
        "bench",
        metrics,
        settings,
        asdict(info),
        environment(),
    )

    backend.close()

    m = {key: fmt(value) for key, value in metrics.items()}
    print(f"{args.rung}: {len(measurements)} generations of {args.max_new_tokens} tokens")
    print(f"  decode  {m['decode_tps_median']} tok/s, e2e {m['e2e_tps_median']} tok/s (medians)")
    print(f"  TTFT    p50 {m['ttft_ms_p50']} ms, p95 {m['ttft_ms_p95']} ms")
    print(f"  ITL     p50 {m['itl_ms_p50']} ms, p99 {m['itl_ms_p99']} ms")
    print(f"  memory  {m['rss_after_load_mb']} MB after load, peak {m['peak_rss_mb']} MB")
    if not consistent:
        print("  WARNING: repeats of the same prompt produced different tokens")
    print(f"  wrote {path}")


def cmd_quality(args: argparse.Namespace) -> None:
    threads = args.threads or default_threads()
    text = load_text_file(args.text_file) if args.text_file else load_wikitext2("test")

    backend = build_backend(args.rung, args.model, threads)
    reference = build_backend(args.reference, args.model, threads) if args.reference else None
    windows = make_windows(backend.encode(text), args.window, args.max_windows)
    result = evaluate_quality(backend, windows, reference)

    settings = {
        "model": args.model,
        "threads": threads,
        "data": args.text_file or "wikitext-2-raw-v1/test",
        "window": args.window,
        "max_windows": args.max_windows,
        "reference": args.reference,
    }
    path = write_result(
        args.out_dir,
        args.machine,
        args.rung,
        "quality",
        asdict(result),
        settings,
        asdict(backend.info),
        environment(),
    )
    backend.close()
    if reference is not None:
        reference.close()

    print(f"{args.rung}: {result.num_windows} windows of {args.window} tokens")
    print(f"  perplexity      {result.perplexity:.3f}")
    if result.top1_agreement is not None:
        print(f"  top-1 agreement {result.top1_agreement:.4%} vs {args.reference}")
        print(f"  mean KL         {result.mean_kl:.3g} nats")
    print(f"  wrote {path}")


def cmd_report(args: argparse.Namespace) -> None:
    path = write_report(args.out_dir, args.machine)
    print(path.read_text(encoding="utf-8"))
    print(f"wrote {path}")


def cmd_rungs(args: argparse.Namespace) -> None:
    for rung in available_rungs():
        print(rung + ("  (baseline)" if rung == BASELINE_RUNG else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ladder", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rung", required=True, choices=available_rungs())
        p.add_argument("--machine", required=True, help="short machine name, e.g. laptop-cpu")
        p.add_argument("--model", default=DEFAULT_MODEL, help="hub id or local model directory")
        p.add_argument("--threads", type=int, help="torch threads (default: physical cores)")
        p.add_argument("--out-dir", default="results")

    bench = sub.add_parser("bench", help="measure speed and memory")
    add_common(bench)
    bench.add_argument("--max-new-tokens", type=int, default=64)
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--prompts", default="data/prompts.jsonl")
    bench.set_defaults(func=cmd_bench)

    quality = sub.add_parser("quality", help="measure perplexity and drift from a reference")
    add_common(quality)
    quality.add_argument("--reference", choices=available_rungs(), help="e.g. " + BASELINE_RUNG)
    quality.add_argument("--window", type=int, default=512)
    quality.add_argument("--max-windows", type=int, help="limit windows for quick runs")
    quality.add_argument("--text-file", help="plain text file instead of WikiText-2 (offline)")
    quality.set_defaults(func=cmd_quality)

    report = sub.add_parser("report", help="write results/<machine>/table.md")
    report.add_argument("--machine", required=True)
    report.add_argument("--out-dir", default="results")
    report.set_defaults(func=cmd_report)

    rungs = sub.add_parser("rungs", help="list available rungs")
    rungs.set_defaults(func=cmd_rungs)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
