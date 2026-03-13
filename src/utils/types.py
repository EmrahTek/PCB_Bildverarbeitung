# dataclasses: BBox, Detection, FrameMeta

"""
types.py

This module defines core data structures used across the PCB component detection project.
The goal is to enforce clear, testable interfaces between the pipeline stages (capture,
preprocessing, detection, postprocessing, rendering).

Key concepts:
- BBox: Axis-aligned bounding box in pixel coordinates.
- Detection: A predicted object instance with label, confidence score, and BBox.
- FrameMeta: Metadata for a captured frame (frame_id, timestamp, source).

Inputs:
- No direct inputs (data classes only).

Outputs:
- Dataclass definitions that are imported by other modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width() * self.height()


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: BBox


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    timestamp_s: float
    source: str
    note: Optional[str] = None
