import json
import time

import numpy as np

from ladder.bench.env import environment
from ladder.bench.memory import RssSampler


def test_rss_sampler_sees_an_allocation():
    with RssSampler(interval_s=0.005) as mem:
        block = np.ones(100 * 2**20, dtype=np.uint8)  # 100 MB, touched by ones()
        time.sleep(0.05)
    assert block[-1] == 1
    assert mem.start_rss_bytes > 0
    assert mem.peak_rss_bytes >= mem.start_rss_bytes + 50 * 2**20


def test_environment_fingerprint():
    env = environment()
    assert {
        "cpu_model",
        "physical_cores",
        "logical_cores",
        "total_ram_mb",
        "os",
        "python",
        "packages",
        "git_commit",
        "timestamp_utc",
    } == set(env)
    assert {"torch", "transformers", "numpy"} <= set(env["packages"])
    assert env["logical_cores"] >= 1
    assert env["timestamp_utc"].endswith("+00:00")
    json.dumps(env)  # must be serialisable as-is
