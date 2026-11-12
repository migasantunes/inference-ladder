import json

import pytest

from ladder.bench.results import read_machine_results, read_result, result_path, write_result
from ladder.report import render_table, write_report


def _bench(out_dir, rung, tps, size_bytes, hashes):
    metrics = {
        "decode_tps_median": tps,
        "e2e_tps_median": tps * 0.9,
        "ttft_ms_p50": 12.0,
        "ttft_ms_p95": 15.0,
        "itl_ms_p50": 1000 / tps,
        "itl_ms_p99": 2000 / tps,
        "rss_after_load_mb": 500.0,
        "peak_rss_mb": 550.0,
        "token_hashes": hashes,
    }
    backend = {"engine": "pytorch", "precision": "fp32", "size_on_disk_bytes": size_bytes}
    return write_result(out_dir, "m1", rung, "bench", metrics, {"threads": 4}, backend, {})


def _quality(out_dir, rung, ppl, agree=None, kl=None):
    metrics = {"perplexity": ppl, "top1_agreement": agree, "mean_kl": kl}
    backend = {"engine": "pytorch", "precision": "fp32", "size_on_disk_bytes": 0}
    return write_result(out_dir, "m1", rung, "quality", metrics, {}, backend, {})


def test_result_round_trip(tmp_path):
    path = write_result(
        tmp_path,
        "m1",
        "01_torch_fp32",
        "bench",
        {"x": 1.5},
        {"threads": 2},
        {"a": "b"},
        {"os": "z"},
    )
    assert path == tmp_path / "m1" / "01_torch_fp32.bench.json"
    assert read_result(path) == {
        "rung": "01_torch_fp32",
        "kind": "bench",
        "machine": "m1",
        "metrics": {"x": 1.5},
        "settings": {"threads": 2},
        "backend": {"a": "b"},
        "environment": {"os": "z"},
    }
    json.loads(path.read_text())  # strict JSON


def test_result_path_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        result_path(tmp_path, "m1", "r", "speed")


def test_report_with_baseline(tmp_path):
    mb = 2**20
    _bench(tmp_path, "01_torch_fp32", 100.0, 400 * mb, {"p": "h1"})
    _quality(tmp_path, "01_torch_fp32", 20.0, 1.0, 0.0)
    _bench(tmp_path, "00_torch_nocache", 25.0, 400 * mb, {"p": "h1"})
    _bench(tmp_path, "03_ort_int8_dynamic", 200.0, 100 * mb, {"p": "DIFFERENT"})
    _quality(tmp_path, "03_ort_int8_dynamic", 21.5, 0.97, 0.012)

    path = write_report(tmp_path, "m1")
    text = path.read_text()
    assert path == tmp_path / "m1" / "table.md"

    lines = [line for line in text.splitlines() if line.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {
        r.split("|")[1].strip(): [c.strip() for c in r.strip("|").split("|")] for r in lines[2:]
    }
    col = {name: i for i, name in enumerate(header)}

    assert list(rows) == ["00_torch_nocache", "01_torch_fp32", "03_ort_int8_dynamic"]
    assert rows["00_torch_nocache"][col["decode speedup"]] == "0.25x"
    assert rows["03_ort_int8_dynamic"][col["decode speedup"]] == "2.00x"
    assert rows["03_ort_int8_dynamic"][col["size reduction"]] == "4.00x"
    assert rows["01_torch_fp32"][col["decode speedup"]] == "1.00x"
    assert rows["00_torch_nocache"][col["tokens = baseline"]] == "yes"
    assert rows["03_ort_int8_dynamic"][col["tokens = baseline"]] == "no"
    assert rows["01_torch_fp32"][col["tokens = baseline"]] == "(baseline)"
    assert rows["03_ort_int8_dynamic"][col["top-1 agreement"]] == "97.0000%"
    assert rows["03_ort_int8_dynamic"][col["perplexity"]] == "21.500"
    # Missing quality run for rung 00 renders as "-".
    assert rows["00_torch_nocache"][col["perplexity"]] == "-"


def test_report_without_baseline_omits_ratios(tmp_path):
    _bench(tmp_path, "00_torch_nocache", 25.0, 0, {"p": "h1"})
    text = render_table(read_machine_results(tmp_path, "m1"))
    assert "speedup" not in text
    assert "reduction" not in text
    header = [c.strip() for c in text.splitlines()[0].strip("|").split("|")]
    row = [c.strip() for c in text.splitlines()[2].strip("|").split("|")]
    assert row[0] == "00_torch_nocache"
    assert row[header.index("decode tok/s")] == "25.0"
    # Size 0 means unknown, so it shows as missing.
    assert row[header.index("size MB")] == "-"
    assert write_report(tmp_path, "m1").read_text().count("ratio columns are omitted") == 1


def test_report_with_no_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_report(tmp_path, "nothing-here")
