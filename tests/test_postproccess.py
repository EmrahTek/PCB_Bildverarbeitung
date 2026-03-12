from __future__ import annotations

from src.detection_logic.postprocess import TemporalDetectionFilter, iou
from src.utils.types import BBox, Detection


def test_iou_is_positive_for_overlapping_boxes() -> None:
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert 0.0 < iou(a, b) < 1.0


def test_temporal_filter_requires_multiple_hits() -> None:
    tf = TemporalDetectionFilter(window_size=3, min_hits=2)
    det = Detection(label="ESP32", score=0.9, bbox=BBox(10, 10, 20, 20))
    assert tf.update([det]) == []
    stable = tf.update([det])
    assert len(stable) == 1
    assert stable[0].label == "ESP32"
