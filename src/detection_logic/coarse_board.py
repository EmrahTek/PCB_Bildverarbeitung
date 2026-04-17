from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.template_match import TemplateMatcher
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class BoardTemplateLocatorConfig:
    """Settings for coarse full-frame board template localisation."""
    resize_width: int = 960
    min_score: float = 0.28


class BoardTemplateLocator:
    """
    Lightweight full-frame board template matcher.

    It returns a coarse board bounding box that can be used as a hint for the
    geometry-based localizer. This is especially helpful when the board is
    attached to a stick or shown in front of cluttered backgrounds.
    """

    def __init__(self, matcher: TemplateMatcher, cfg: BoardTemplateLocatorConfig = BoardTemplateLocatorConfig()) -> None:
        self._matcher = matcher
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> Detection | None:
        search_frame, scale = self._resize_for_search(frame)
        detection = self._matcher.detect_best(search_frame)
        if detection is None or detection.score < self._cfg.min_score:
            return None

        if scale == 1.0:
            return detection

        return Detection(
            label=detection.label,
            score=detection.score,
            bbox=BBox(
                x1=int(round(detection.bbox.x1 / scale)),
                y1=int(round(detection.bbox.y1 / scale)),
                x2=int(round(detection.bbox.x2 / scale)),
                y2=int(round(detection.bbox.y2 / scale)),
            ),
        )

    def _resize_for_search(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        target_width = self._cfg.resize_width
        if target_width <= 0 or width <= target_width:
            return frame, 1.0

        scale = target_width / float(width)
        resized = cv.resize(frame, (target_width, int(round(height * scale))), interpolation=cv.INTER_AREA)
        return resized, scale
