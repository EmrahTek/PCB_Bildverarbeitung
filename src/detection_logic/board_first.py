from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.coarse_board import BoardTemplateLocator
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
    layout_roi: RelativeROI | None = None
    layout_fallback_score: float = 0.0
    layout_fallback_min_board_score: float = 0.60
    min_board_area_ratio: float = 0.0
    max_board_area_ratio: float = 1.0
    min_normalized_aspect_ratio: float = 1.0
    max_normalized_aspect_ratio: float = 10.0
    min_board_overlap_ratio: float = 0.85


@dataclass(frozen=True)
class BoardFirstConfig:
    """Stateful detector settings for webcam usage."""
    temporal_window: int = 5
    temporal_min_hits: int = 2
    max_missing_frames: int = 4
    template_refresh_interval: int = 5
    hint_accept_score: float = 0.58
    enable_tracking: bool = True


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
        board_locator: BoardTemplateLocator | None = None,
        cfg: BoardFirstConfig = BoardFirstConfig(),
    ) -> None:
        self._localizer = localizer
        self._component_matchers = component_matchers
        self._component_specs = component_specs
        self._board_locator = board_locator
        self._cfg = cfg
        self._temporal = TemporalDetectionFilter(window_size=cfg.temporal_window, min_hits=cfg.temporal_min_hits)
        self._last_board_bbox: BBox | None = None
        self._last_coarse_bbox: BBox | None = None
        self._last_coarse_bboxes: list[BBox] = []
        self._missing_frames = 0
        self._frame_index = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frame_index += 1

        coarse_hints = self._maybe_refresh_coarse_hints(frame)
        tracking_hint = self._last_board_bbox if self._cfg.enable_tracking else None

        localization = self._localize_from_hints(frame, coarse_hints, tracking_hint)

        if localization is None:
            self._missing_frames += 1
            if self._missing_frames > self._cfg.max_missing_frames:
                self._last_board_bbox = None
                self._last_coarse_bbox = None
                self._last_coarse_bboxes = []
            return self._temporal.update([])

        if self._cfg.enable_tracking:
            self._last_board_bbox = localization.bbox
        else:
            self._last_board_bbox = None
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
                fallback = self._layout_fallback_detection(spec, localization, frame_shape)
                if fallback is not None:
                    out.append(fallback)
                continue

            canonical_bbox = BBox(
                roi_bbox.x1 + detection.bbox.x1,
                roi_bbox.y1 + detection.bbox.y1,
                roi_bbox.x1 + detection.bbox.x2,
                roi_bbox.y1 + detection.bbox.y2,
            )
            mapped_bbox = map_bbox_with_homography(canonical_bbox, localization.h_inv, frame_shape)
            if mapped_bbox is None or mapped_bbox.area() <= 0:
                fallback = self._layout_fallback_detection(spec, localization, frame_shape)
                if fallback is not None:
                    out.append(fallback)
                continue
            if not self._passes_component_sanity(spec, mapped_bbox, localization.bbox):
                fallback = self._layout_fallback_detection(spec, localization, frame_shape)
                if fallback is not None:
                    out.append(fallback)
                continue

            out.append(Detection(label=spec.label, score=detection.score, bbox=mapped_bbox))
        return out

    def _layout_fallback_detection(
        self,
        spec: ComponentSpec,
        localization: BoardLocalization,
        frame_shape: tuple[int, ...],
    ) -> Detection | None:
        if spec.layout_roi is None or spec.layout_fallback_score <= 0.0:
            return None
        if localization.score < spec.layout_fallback_min_board_score:
            return None

        canonical_bbox = self._roi_to_bbox(spec.layout_roi, localization.warped.shape)
        mapped_bbox = map_bbox_with_homography(canonical_bbox, localization.h_inv, frame_shape)
        if mapped_bbox is None or mapped_bbox.area() <= 0:
            return None
        if not self._passes_component_sanity(spec, mapped_bbox, localization.bbox):
            return None

        score = min(float(localization.score), float(spec.layout_fallback_score))
        return Detection(label=spec.label, score=score, bbox=mapped_bbox)

    @staticmethod
    def _roi_to_bbox(roi: RelativeROI, shape: tuple[int, ...]) -> BBox:
        height, width = shape[:2]
        x1 = int(round(roi.x1f * width))
        y1 = int(round(roi.y1f * height))
        x2 = int(round(roi.x2f * width))
        y2 = int(round(roi.y2f * height))
        return BBox(x1, y1, x2, y2)

    @staticmethod
    def _passes_component_sanity(spec: ComponentSpec, component_bbox: BBox, board_bbox: BBox) -> bool:
        board_area = max(1, board_bbox.area())
        component_area_ratio = component_bbox.area() / board_area
        if component_area_ratio < spec.min_board_area_ratio or component_area_ratio > spec.max_board_area_ratio:
            return False

        width = max(1, component_bbox.width())
        height = max(1, component_bbox.height())
        aspect = width / height
        normalized_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
        if (
            normalized_aspect < spec.min_normalized_aspect_ratio
            or normalized_aspect > spec.max_normalized_aspect_ratio
        ):
            return False

        overlap_ratio = BoardFirstDetector._overlap_ratio(component_bbox, board_bbox)
        if overlap_ratio < spec.min_board_overlap_ratio:
            return False

        return True

    @staticmethod
    def _overlap_ratio(inner: BBox, outer: BBox) -> float:
        inter_x1 = max(inner.x1, outer.x1)
        inter_y1 = max(inner.y1, outer.y1)
        inter_x2 = min(inner.x2, outer.x2)
        inter_y2 = min(inner.y2, outer.y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        return inter_area / max(1, inner.area())

    def _localize_from_hints(
        self,
        frame: np.ndarray,
        coarse_hints: list[BBox],
        tracking_hint: BBox | None,
    ) -> BoardLocalization | None:
        hints: list[BBox] = []
        hints.extend(coarse_hints)
        if tracking_hint is not None:
            hints.append(tracking_hint)

        unique_hints: list[BBox] = []
        seen: set[tuple[int, int, int, int]] = set()
        for hint in hints:
            key = (hint.x1 // 8, hint.y1 // 8, hint.x2 // 8, hint.y2 // 8)
            if key in seen:
                continue
            seen.add(key)
            unique_hints.append(hint)

        best: BoardLocalization | None = None
        for hint in unique_hints[:4]:
            candidate = self._localizer.localize(frame, hint_bbox=hint, include_full_frame=False)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None and best.score >= self._cfg.hint_accept_score:
            return best

        full_frame = self._localizer.localize(frame)
        if full_frame is not None and (best is None or full_frame.score > best.score):
            best = full_frame
        return best

    def _maybe_refresh_coarse_hints(self, frame: np.ndarray) -> list[BBox]:
        if self._board_locator is None:
            return list(self._last_coarse_bboxes)

        should_refresh = (
            self._last_coarse_bbox is None
            or not self._cfg.enable_tracking
            or self._frame_index % max(1, self._cfg.template_refresh_interval) == 0
        )
        if should_refresh:
            if hasattr(self._board_locator, "detect_candidates"):
                coarse_candidates = self._board_locator.detect_candidates(frame)
            else:
                coarse = self._board_locator.detect(frame)
                coarse_candidates = [coarse] if coarse is not None else []
            self._last_coarse_bboxes = [candidate.bbox for candidate in coarse_candidates if candidate is not None]
            self._last_coarse_bbox = self._last_coarse_bboxes[0] if self._last_coarse_bboxes else None
        return list(self._last_coarse_bboxes)
