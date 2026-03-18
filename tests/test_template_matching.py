from __future__ import annotations

import cv2 as cv
import numpy as np

from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher


def test_template_matcher_detects_simple_rectangle() -> None:
    frame = np.zeros((160, 240), dtype=np.uint8)
    cv.rectangle(frame, (80, 40), (140, 90), 255, thickness=-1)

    template = np.zeros((50, 60), dtype=np.uint8)
    cv.rectangle(template, (0, 0), (59, 49), 255, thickness=-1)

    matcher = TemplateMatcher(
        [template],
        TemplateMatchConfig(
            label="RECT",
            score_threshold=0.80,
            scales=(1.0,),
            top_k=1,
            use_clahe=False,
            edge_mode=False,
            blur_ksize=3,
        ),
    )

    detections = matcher.detect(frame)
    assert len(detections) == 1
    assert detections[0].label == "RECT"
