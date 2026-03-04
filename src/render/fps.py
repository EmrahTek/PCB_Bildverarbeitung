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
from collections import deque # Double Ended Queue

class FPSCounter:
    """
    Simple FPS estimator using a sliding window of timestamps.
    """
    def __init__(self,window_size, int= 30) -> None:
        if window_size <= 1:
            raise ValueError("Window_size must be > 1")
        self._ts = deque(maxlen=window_size)
    def tick(self) -> float:
        """
        Register a new frame timestamp and return estimated FPS.
        """
        now = time.perf_counter()
        self._ts.append(now)

        if len(self._ts) < 2:
            return 0.0
        
        dt = self._ts[-1] - self._ts[0]
        if dt <= 0:
            return 0.0
        # (N-1) intervals over total duration dt
        return (len(self._ts) - 1) / dt
    

