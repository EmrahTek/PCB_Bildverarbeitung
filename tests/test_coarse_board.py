from __future__ import annotations

import numpy as np

from src.detection_logic.coarse_board import BoardTemplateLocator, BoardTemplateLocatorConfig
from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher


def test_board_template_locator_maps_bbox_back_to_original_scale() -> None:
    template = np.zeros((40, 80), dtype=np.uint8)
    template[6:34, 8:72] = 220

    frame = np.zeros((240, 480), dtype=np.uint8)
    frame[100:140, 220:300] = template

    matcher = TemplateMatcher(
        [template],
        TemplateMatchConfig(
            label="BOARD",
            score_threshold=0.20,
            scales=(1.0,),
            gray_weight=1.0,
            edge_weight=0.0,
            use_clahe=False,
            blur_ksize=1,
            min_template_size=8,
        ),
    )
    locator = BoardTemplateLocator(matcher, BoardTemplateLocatorConfig(resize_width=240, min_score=0.20))

    detection = locator.detect(frame)
    assert detection is not None
    assert detection.bbox.x1 <= 220 <= detection.bbox.x2
    assert detection.bbox.y1 <= 100 <= detection.bbox.y2
