"""Fingerprint of the machine and software a result was produced on.

Stored with every result so numbers from different machines or library versions
are never mixed by accident.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import psutil

# Library versions are read from package metadata, so torch or onnxruntime are
# not imported just to ask for their version.
TRACKED_PACKAGES = ("torch", "transformers", "numpy", "onnxruntime")


def cpu_model() -> str:
    """Human-readable CPU name. ``platform.processor()`` is often empty on Linux."""
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def git_commit() -> str | None:
    """Commit hash of the checkout this code runs from, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def package_versions() -> dict[str, str]:
    """Installed versions of the tracked libraries; missing ones are left out."""
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def environment() -> dict[str, object]:
    """Everything recorded about the machine and software for one run."""
    return {
        "cpu_model": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_mb": round(psutil.virtual_memory().total / 2**20),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
