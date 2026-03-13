# FPS Counter
# Stabiler FPS-Tracker (gleitender Mittelwert), damit ihr Performance messen könnt.

"""
fps.py

This module provides an FPS tracker to estimate frames-per-second in a stable way.
It can be based on a rolling window to smooth fluctuations.

Inputs:
- tick() calls per processed frame

Outputs:
- Current FPS estimate as float


time module:
https://docs.python.org/3/library/time.html

Zu implementierende Funktionen / Klassen

class FPSTracker:

        tick() -> None

        fps() -> float

        (Optional) reset()

"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Estimate FPS using a sliding window of timestamps."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt
