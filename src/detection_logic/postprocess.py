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

