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