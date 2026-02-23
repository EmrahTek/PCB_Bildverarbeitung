# FrameSource Interface
# Gemeinsames Interface für Detektoren (Template Matching jetzt; ORB optional später).

"""
base.py

This module defines the base interface for frame sources.
A frame source abstracts away where frames come from (webcam, video file, etc.)
to keep the processing pipeline testable and deterministic.

Inputs:
- Implementation-specific parameters (e.g., camera index, video file path).

Outputs:
- A unified API: open(), read() -> (frame, meta), close()

Same resources as base interfaces:
https://docs.python.org/3/library/abc.html
https://docs.python.org/3/library/typing.html#typing.Protocol

Zu implementierende Funktionen / Klassen

class Detector(Protocol/ABC):

detect(image: np.ndarray) -> list[Detection]

"""

