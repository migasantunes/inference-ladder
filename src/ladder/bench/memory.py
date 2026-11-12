"""Resident memory (RSS) of this process, sampled in a background thread.

A single reading at the end would miss short spikes during a run, so a thread
reads RSS every few milliseconds and keeps the maximum. psutil makes this
work the same way on Linux, macOS and Windows.
"""

from __future__ import annotations

import threading

import psutil


def current_rss_bytes() -> int:
    """Resident set size of this process right now."""
    return psutil.Process().memory_info().rss


class RssSampler:
    """Context manager that records RSS at start and the peak while it runs.

    Usage::

        with RssSampler() as mem:
            run_benchmark()
        print(mem.start_rss_bytes, mem.peak_rss_bytes)
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss_bytes())

    def _run(self) -> None:
        # Event.wait doubles as the sleep between samples and the stop signal.
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> RssSampler:
        self.start_rss_bytes = current_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()  # one last reading, in case the peak came at the very end
