from __future__ import annotations

import numpy as np

from src.app.pipeline import Pipeline
from src.camera_input.base import FrameSource
from src.utils.types import BBox, Detection, FrameMeta


class DummySource(FrameSource):
    def __init__(self) -> None:
        self._count = 0

    def open(self) -> None:
        return None

    def read(self):
        if self._count >= 3:
            return None, None
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        meta = FrameMeta(frame_id=self._count, timestamp_s=0.0, source="dummy")
        self._count += 1
        return frame, meta

    def release(self) -> None:
        return None


class DummyDetector:
    def detect(self, frame: np.ndarray):
        return [Detection(label="ESP32", score=0.95, bbox=BBox(10, 10, 50, 40))]


def test_pipeline_runs_headless_without_crash() -> None:
    pipeline = Pipeline(detector=DummyDetector())
    pipeline.run(DummySource(), debug=False, headless=True, max_frames=3, wait_ms=0)
