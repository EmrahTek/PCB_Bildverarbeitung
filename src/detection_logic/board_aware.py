from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.feature_match import ORBFeatureMatcher
from src.detection_logic.postprocess import TemporalDetectionFilter, map_detections_with_homography
from src.detection_logic.template_match import TemplateMatcher
from src.preprocessing.geometry import BoardLocalization, BoardLocalizer
from src.utils.types import Detection


@dataclass(frozen=True)
class BoardAwareConfig:
    allow_full_frame_fallback: bool = True


class BoardAwareHybridDetector(Detector):
    """Detect components on a warped board and map results back to the original frame."""

    def __init__(
        self,
        *,
        localizer: BoardLocalizer,
        primary_detectors: dict[str, TemplateMatcher],
        fallback_detectors: dict[str, ORBFeatureMatcher] | None = None,
        allow_full_frame_fallback: bool = True,
        temporal_window: int = 5,
        temporal_min_hits: int = 3,
    ) -> None:
        if not primary_detectors:
            raise ValueError("primary_detectors must not be empty")
        self._localizer = localizer
        self._primary_detectors = primary_detectors
        self._fallback_detectors = fallback_detectors or {}
        self._allow_full_frame_fallback = allow_full_frame_fallback
        self._temporal = TemporalDetectionFilter(window_size=temporal_window, min_hits=temporal_min_hits)
        self._frame_id = 0
        self.last_board_quad: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frame_id += 1
        localization = self._localizer.localize(frame, self._frame_id)
        self.last_board_quad = localization.quad if localization is not None else None

        detections: list[Detection] = []
        if localization is not None:
            detections = self._detect_on_warped_board(localization)
        elif self._allow_full_frame_fallback:
            detections = self._detect_on_full_frame(frame)

        return self._temporal.update(detections)

    def _detect_on_warped_board(self, localization: BoardLocalization) -> list[Detection]:
        warped = localization.warped
        all_detections: list[Detection] = []
        for label, detector in self._primary_detectors.items():
            dets = detector.detect(warped)
            if not dets and label in self._fallback_detectors:
                dets = self._fallback_detectors[label].detect(warped)
            all_detections.extend(map_detections_with_homography(dets, localization.h_inv))
        return all_detections

    def _detect_on_full_frame(self, frame: np.ndarray) -> list[Detection]:
        all_detections: list[Detection] = []
        for label, detector in self._primary_detectors.items():
            dets = detector.detect(frame)
            if not dets and label in self._fallback_detectors:
                dets = self._fallback_detectors[label].detect(frame)
            all_detections.extend(dets)
        return all_detections
