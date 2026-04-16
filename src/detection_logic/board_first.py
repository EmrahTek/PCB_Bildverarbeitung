from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import TemporalDetectionFilter, map_bbox_with_homography
from src.detection_logic.template_match import TemplateMatcher
from src.preprocessing.geometry import BoardLocalization, BoardLocalizer
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class RelativeROI:
    """Component search region in canonical board coordinates."""
    x1f: float
    y1f: float
    x2f: float
    y2f: float


@dataclass(frozen=True)
class ComponentSpec:
    """A single board component definition."""
    label: str
    roi: RelativeROI
    score_threshold: float


@dataclass(frozen=True)
class BoardFirstConfig:
    """Stateful detector settings for webcam usage."""
    temporal_window: int = 5
    temporal_min_hits: int = 2
    max_missing_frames: int = 4


class BoardFirstDetector(Detector):
    """
    Detect the board first, then search components inside fixed canonical ROIs.

    This is the main detector used by the application. It keeps a small amount of
    state so webcam usage remains stable while the board is being moved by hand.
    """

    def __init__(
        self,
        localizer: BoardLocalizer,
        component_matchers: dict[str, TemplateMatcher],
        component_specs: list[ComponentSpec],
        cfg: BoardFirstConfig = BoardFirstConfig(),
    ) -> None:
        self._localizer = localizer
        self._component_matchers = component_matchers
        self._component_specs = component_specs
        self._cfg = cfg
        self._temporal = TemporalDetectionFilter(window_size=cfg.temporal_window, min_hits=cfg.temporal_min_hits)
        self._last_board_bbox: BBox | None = None
        self._missing_frames = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        localization = self._localizer.localize(frame, hint_bbox=self._last_board_bbox)
        if localization is None:
            self._missing_frames += 1
            if self._missing_frames > self._cfg.max_missing_frames:
                self._last_board_bbox = None
            return self._temporal.update([])

        self._last_board_bbox = localization.bbox
        self._missing_frames = 0

        detections = [Detection(label="BOARD", score=localization.score, bbox=localization.bbox)]
        detections.extend(self._detect_components(localization, frame.shape))
        return self._temporal.update(detections)

    def _detect_components(self, localization: BoardLocalization, frame_shape: tuple[int, ...]) -> list[Detection]:
        out: list[Detection] = []
        for spec in self._component_specs:
            matcher = self._component_matchers.get(spec.label)
            if matcher is None:
                continue

            roi_bbox = self._roi_to_bbox(spec.roi, localization.warped.shape)
            crop = localization.warped[roi_bbox.y1:roi_bbox.y2, roi_bbox.x1:roi_bbox.x2]
            if crop.size == 0:
                continue

            detection = matcher.detect_best(crop)
            if detection is None or detection.score < spec.score_threshold:
                continue

            canonical_bbox = BBox(
                roi_bbox.x1 + detection.bbox.x1,
                roi_bbox.y1 + detection.bbox.y1,
                roi_bbox.x1 + detection.bbox.x2,
                roi_bbox.y1 + detection.bbox.y2,
            )
            mapped_bbox = map_bbox_with_homography(canonical_bbox, localization.h_inv, frame_shape)
            if mapped_bbox is None or mapped_bbox.area() <= 0:
                continue

            out.append(Detection(label=spec.label, score=detection.score, bbox=mapped_bbox))
        return out

    @staticmethod
    def _roi_to_bbox(roi: RelativeROI, shape: tuple[int, ...]) -> BBox:
        height, width = shape[:2]
        x1 = int(round(roi.x1f * width))
        y1 = int(round(roi.y1f * height))
        x2 = int(round(roi.x2f * width))
        y2 = int(round(roi.y2f * height))
        return BBox(x1, y1, x2, y2)
