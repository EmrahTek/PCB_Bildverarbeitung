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
from typing import List
from src.utils.types import BBox, Detection

def iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two bounding boxes."""
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
        detections:List[Detection],
        *,
        iou_threshold: float = 0.3,
        max_detections: int | None = None,
)-> List[Detection]:
    """
    Standard greedy NMS.

    Keeps highest-score detections and suppresses overlapping ones.
    """
    if not detections:
        return []
    
    candidates = sorted(detections, key = lambda d: d.score, reverse=True)

    kept: list[Detection] = []
    for det in candidates:
        if all(iou(det.bbox, k.bbox ) < iou_threshold for k in kept):
            kept.append(det)
            if max_detections is not None and len(kept) >= max_detections:
                break
    return kept

