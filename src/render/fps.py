"""FPS estimation for live display and diagnostics.

This module keeps a small sliding window of frame timestamps and reports a
smoothed frames-per-second estimate.

Python docs:
- collections.deque: https://docs.python.org/3/library/collections.html#collections.deque
- time: https://docs.python.org/3/library/time.html
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Estimate FPS using a sliding time window."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        delta_t = self._timestamps[-1] - self._timestamps[0]
        if delta_t <= 0.0:
            return 0.0
        return (len(self._timestamps) - 1) / delta_t
