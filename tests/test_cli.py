"""End to end: the real CLI against the tiny model saved to disk. No network."""

import json
import subprocess
import sys

import pytest

from ladder.cli import main
from ladder.registry import available_rungs, build_backend

SPEED_KEYS = {
    "ttft_ms_p50",
    "ttft_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "itl_ms_p99",
    "decode_tps_median",
    "e2e_tps_median",
    "load_time_s",
    "rss_after_load_mb",
    "peak_rss_mb",
    "size_on_disk_mb",
    "token_hashes",
    "tokens_consistent_across_repeats",
}
RECORD_KEYS = {"rung", "kind", "machine", "metrics", "settings", "backend", "environment"}


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [{"id": "a", "text": "The quick brown fox"}, {"id": "b", "text": "A model predicts"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _bench(model_dir, out_dir, prompts, rung):
    main(
        ["bench", "--rung", rung, "--machine", "tiny", "--model", str(model_dir)]
        + ["--threads", "1", "--max-new-tokens", "8", "--warmup", "1", "--repeats", "2"]
        + ["--prompts", str(prompts), "--out-dir", str(out_dir)]
    )


def test_bench_quality_report_end_to_end(tiny_model_dir, tmp_path, prompts_file, capsys):
    out = tmp_path / "results"
    _bench(tiny_model_dir, out, prompts_file, "01_torch_fp32")
    _bench(tiny_model_dir, out, prompts_file, "00_torch_nocache")

    text_file = tmp_path / "text.txt"
    text_file.write_text("The quick brown fox jumps over the lazy dog.\n" * 20)
    main(
        ["quality", "--rung", "00_torch_nocache", "--machine", "tiny"]
        + ["--model", str(tiny_model_dir), "--reference", "01_torch_fp32"]
        + ["--window", "32", "--max-windows", "4", "--text-file", str(text_file)]
        + ["--threads", "1", "--out-dir", str(out)]
    )
    main(["report", "--machine", "tiny", "--out-dir", str(out)])

    bench = json.loads((out / "tiny" / "01_torch_fp32.bench.json").read_text())
    assert set(bench) == RECORD_KEYS
    assert SPEED_KEYS <= set(bench["metrics"])
    assert bench["settings"]["num_generations"] == 4
    assert bench["backend"]["size_on_disk_bytes"] > 0
    assert {"cpu_model", "packages", "git_commit", "timestamp_utc"} <= set(bench["environment"])

    quality = json.loads((out / "tiny" / "00_torch_nocache.quality.json").read_text())
    assert set(quality) == RECORD_KEYS
    assert quality["metrics"]["num_windows"] == 4
    assert quality["metrics"]["top1_agreement"] == 1.0  # the cache does not change results
    assert quality["metrics"]["mean_kl"] == pytest.approx(0, abs=1e-12)

    table = (out / "tiny" / "table.md").read_text()
    assert "| 00_torch_nocache |" in table
    assert "| 01_torch_fp32 |" in table
    assert "decode speedup" in table
    nocache_row = next(line for line in table.splitlines() if "00_torch_nocache" in line)
    assert nocache_row.rstrip().endswith("| yes |")  # same tokens as the baseline
    assert "table.md" in capsys.readouterr().out


def test_rungs_command(capsys):
    main(["rungs"])
    out = capsys.readouterr().out
    assert "00_torch_nocache" in out
    assert "01_torch_fp32  (baseline)" in out


def test_registry_unknown_rung():
    assert available_rungs() == ["00_torch_nocache", "01_torch_fp32"]
    with pytest.raises(ValueError, match="Available rungs: 00_torch_nocache, 01_torch_fp32"):
        build_backend("99_magic", "model", None)


def test_importing_cli_stays_light():
    code = (
        "import sys, ladder, ladder.cli; "
        "heavy = ['datasets', 'onnxruntime', 'fastapi', 'matplotlib', 'torch', 'transformers']; "
        "print([m for m in heavy if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
