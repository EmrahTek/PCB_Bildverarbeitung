from __future__ import annotations

import numpy as np

from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher


def test_template_matcher_detects_embedded_patch() -> None:
    template = np.zeros((30, 40), dtype=np.uint8)
    template[4:26, 6:34] = 220

    scene = np.zeros((120, 160), dtype=np.uint8)
    scene[55:85, 70:110] = template

    matcher = TemplateMatcher(
        [template],
        TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.30,
            scales=(1.0,),
            gray_weight=1.0,
            edge_weight=0.0,
            use_clahe=False,
            blur_ksize=1,
            min_template_size=8,
        ),
    )

    detection = matcher.detect_best(scene)
    assert detection is not None
    assert detection.bbox.x1 <= 70 <= detection.bbox.x2
    assert detection.bbox.y1 <= 55 <= detection.bbox.y2
    assert detection.score >= 0.30


def test_template_matcher_returns_multiple_candidates() -> None:
    template = np.zeros((20, 30), dtype=np.uint8)
    template[4:16, 5:25] = 220

    scene = np.zeros((120, 180), dtype=np.uint8)
    scene[20:40, 30:60] = template
    scene[75:95, 120:150] = template

    matcher = TemplateMatcher(
        [template],
        TemplateMatchConfig(
            label="BOARD",
            score_threshold=0.30,
            scales=(1.0,),
            gray_weight=1.0,
            edge_weight=0.0,
            use_clahe=False,
            blur_ksize=1,
            min_template_size=8,
        ),
    )

    detections = matcher.detect_candidates(scene, max_candidates=2, score_threshold=0.30)
    assert len(detections) == 2
    assert detections[0].score >= 0.30
    assert detections[1].score >= 0.30
