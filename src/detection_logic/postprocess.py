# NMS, Thresholding, Score-Filter
# Treffer bereinigen: Score-Filter, Non-Maximum Suppression (NMS), Counts pro Label.

"""
postprocess.py

This module contains postprocessing utilities for raw detections:
- score filtering
- Non-Maximum Suppression (NMS) using IoU
- counting detections per label

Inputs:
- list[Detection] from a detector

Outputs:
- list[Detection] after filtering/NMS
- dict[label, count] for UI overlay and logging

Zu implementierende Funktionen

    filter_by_score(detections, min_score) -> detections

    iou(a: BBox, b: BBox) -> float

    non_max_suppression(detections, iou_threshold) -> detections

    count_by_label(detections) -> dict[str, int]

Intersection over Union (search terms):
"IoU bounding box explanation"
"non maximum suppression implementation python"

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


def nms_detections(
    detections: list[Detection],
    *,
    iou_threshold: float = 0.3,
    max_detections: int | None = None,
) -> list[Detection]:
    """Greedy non-maximum suppression based on score ordering."""
    if not detections:
        return []

    candidates = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []

    for det in candidates:
        if all(iou(det.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(det)
            if max_detections is not None and len(kept) >= max_detections:
                break
    return kept


def count_by_label(detections: Iterable[Detection]) -> dict[str, int]:
    return dict(Counter(det.label for det in detections))


def translate_detections(detections: Iterable[Detection], dx: int, dy: int) -> list[Detection]:
    """Translate detections from an ROI crop back into its parent image coordinates."""
    out: list[Detection] = []
    for det in detections:
        box = det.bbox
        out.append(
            Detection(
                label=det.label,
                score=det.score,
                bbox=BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy),
            )
        )
    return out


def map_detections_with_homography(detections: Iterable[Detection], h_inv: np.ndarray) -> list[Detection]:
    """Map detections from warped-board space back to original frame coordinates."""
    out: list[Detection] = []
    for det in detections:
        box = det.bbox
        corners = np.array(
            [[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mapped = cv.perspectiveTransform(corners, h_inv).reshape(-1, 2)
        x1, y1 = mapped.min(axis=0)
        x2, y2 = mapped.max(axis=0)
        out.append(
            Detection(
                label=det.label,
                score=det.score,
                bbox=BBox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )
    return out


@dataclass
class TemporalDetectionFilter:
    """Simple temporal voting filter to suppress one-frame false positives."""

    window_size: int = 5
    min_hits: int = 3

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.min_hits <= 0:
            raise ValueError("min_hits must be positive")
        self._history: dict[str, deque[Detection | None]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(self, detections: list[Detection]) -> list[Detection]:
        grouped: dict[str, Detection] = {}
        for det in detections:
            best = grouped.get(det.label)
            if best is None or det.score > best.score:
                grouped[det.label] = det

        labels = set(self._history.keys()) | set(grouped.keys())
        for label in labels:
            self._history[label].append(grouped.get(label))

        stable: list[Detection] = []
        for label, hist in self._history.items():
            present = [item for item in hist if item is not None]
            if len(present) < self.min_hits:
                continue
            x1 = int(round(sum(d.bbox.x1 for d in present) / len(present)))
            y1 = int(round(sum(d.bbox.y1 for d in present) / len(present)))
            x2 = int(round(sum(d.bbox.x2 for d in present) / len(present)))
            y2 = int(round(sum(d.bbox.y2 for d in present) / len(present)))
            score = max(d.score for d in present)
            stable.append(Detection(label=label, score=score, bbox=BBox(x1, y1, x2, y2)))

        return stable
