"""Postprocessing helpers for detection boxes.

This module computes overlap metrics, maps canonical board boxes back to the
original frame, counts labels, and smooths detections over short time windows.

Python docs:
- collections: https://docs.python.org/3/library/collections.html
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- typing: https://docs.python.org/3/library/typing.html
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from src.utils.types import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    """Compute intersection over union for two axis-aligned boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = a.area() + b.area() - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def count_by_label(detections: Iterable[Detection]) -> dict[str, int]:
    """Count detections per label."""
    return dict(Counter(det.label for det in detections))


def map_bbox_with_homography(bbox: BBox, h_inv: np.ndarray, frame_shape: tuple[int, ...]) -> BBox | None:
    """Map a bounding box from canonical board space back to original image space."""
    points = np.array(
        [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    mapped = cv.perspectiveTransform(points, h_inv).reshape(-1, 2)
    h, w = frame_shape[:2]
    x1 = int(max(0, np.floor(mapped[:, 0].min())))
    y1 = int(max(0, np.floor(mapped[:, 1].min())))
    x2 = int(min(w - 1, np.ceil(mapped[:, 0].max())))
    y2 = int(min(h - 1, np.ceil(mapped[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2)


@dataclass
class TemporalDetectionFilter:
    """
    Simple temporal voting filter.

    The filter keeps one best detection per label per frame and only emits a stable
    result once a label appears often enough inside the sliding time window.
    """

    window_size: int = 5
    min_hits: int = 2

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_hits <= 0:
            raise ValueError("min_hits must be positive")
        self._history: dict[str, deque[Detection | None]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def reset(self) -> None:
        """Clear all temporal history after the tracked board is lost."""
        self._history.clear()

    def update(self, detections: list[Detection]) -> list[Detection]:
        best_per_label: dict[str, Detection] = {}
        for detection in detections:
            current = best_per_label.get(detection.label)
            if current is None or detection.score > current.score:
                best_per_label[detection.label] = detection

        labels = set(self._history.keys()) | set(best_per_label.keys())
        for label in labels:
            self._history[label].append(best_per_label.get(label))

        stable: list[Detection] = []
        for label, history in self._history.items():
            present = [item for item in history if item is not None]
            if len(present) < self.min_hits:
                continue
            x1 = int(round(sum(det.bbox.x1 for det in present) / len(present)))
            y1 = int(round(sum(det.bbox.y1 for det in present) / len(present)))
            x2 = int(round(sum(det.bbox.x2 for det in present) / len(present)))
            y2 = int(round(sum(det.bbox.y2 for det in present) / len(present)))
            score = max(det.score for det in present)
            stable.append(Detection(label=label, score=score, bbox=BBox(x1, y1, x2, y2)))

        stable.sort(key=lambda det: det.label)
        return stable
