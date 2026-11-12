"""Result files: ``<out_dir>/<machine>/<rung>.<kind>.json``.

Every file has the same top-level shape::

    {
      "rung": "01_torch_fp32",
      "kind": "bench" | "quality",
      "machine": "laptop-cpu",
      "metrics": {...},       # the numbers the report shows
      "settings": {...},      # the options the run used
      "backend": {...},       # BackendInfo of the loaded backend
      "environment": {...}    # see ladder.bench.env
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KINDS = ("bench", "quality")


def result_path(out_dir: str | Path, machine: str, rung: str, kind: str) -> Path:
    """Where the result for one rung and kind lives."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return Path(out_dir) / machine / f"{rung}.{kind}.json"


def write_result(
    out_dir: str | Path,
    machine: str,
    rung: str,
    kind: str,
    metrics: dict[str, Any],
    settings: dict[str, Any],
    backend: dict[str, Any],
    environment: dict[str, Any],
) -> Path:
    """Write one result file, replacing any earlier run of the same rung and kind."""
    path = result_path(out_dir, machine, rung, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "rung": rung,
        "kind": kind,
        "machine": machine,
        "metrics": metrics,
        "settings": settings,
        "backend": backend,
        "environment": environment,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_result(path: str | Path) -> dict[str, Any]:
    """Load one result file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_machine_results(out_dir: str | Path, machine: str) -> dict[str, dict[str, dict]]:
    """All results for a machine, as ``{rung: {kind: record}}``."""
    results: dict[str, dict[str, dict]] = {}
    for path in sorted((Path(out_dir) / machine).glob("*.json")):
        record = read_result(path)
        results.setdefault(record["rung"], {})[record["kind"]] = record
    return results
