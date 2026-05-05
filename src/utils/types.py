"""Core dataclasses shared by the PCB detection project.

These small value objects define the common data passed between capture,
preprocessing, detection, postprocessing, logging, and rendering stages.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- typing.Optional: https://docs.python.org/3/library/typing.html#typing.Optional
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
