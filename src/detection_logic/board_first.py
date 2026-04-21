from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.coarse_board import BoardTemplateLocator
from src.detection_logic.postprocess import TemporalDetectionFilter, map_bbox_with_homography
from src.detection_logic.template_match import TemplateMatcher, TemplateMatchResult
from src.preprocessing.geometry import BoardLocalization, BoardLocalizer
from src.utils.types import BBox, Detection

LOGGER = logging.getLogger(__name__)


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
    min_warp_quality_score: float = 0.0
    layout_fallback_min_warp_quality: float = 0.0
    layout_fallback_min_match_score: float = 0.0
    min_visibility_score: float = 0.0
    layout_fallback_min_visibility_score: float = 0.0
    visibility_weight: float = 0.0
    warp_quality_weight: float = 0.0


@dataclass(frozen=True)
class BoardFirstConfig:
    """Stateful detector settings for webcam usage."""
    temporal_window: int = 5
    temporal_min_hits: int = 2
    max_missing_frames: int = 4
    template_refresh_interval: int = 5
    hint_accept_score: float = 0.58
    enable_tracking: bool = True
    max_pose_reuse_frames: int = 2
    reuse_min_warp_quality: float = 0.48
    reuse_score_decay: float = 0.92


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
        self._last_localization: BoardLocalization | None = None
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
            reused = self._reuse_last_localization(frame)
            if reused is not None:
                LOGGER.debug(
                    "reusing previous board pose: missing_frames=%d score=%.3f warp_quality=%.3f",
                    self._missing_frames,
                    reused.score,
                    reused.warp_quality_score,
                )
                detections = [Detection(label="BOARD", score=reused.score, bbox=reused.bbox)]
                detections.extend(self._detect_components(reused, frame.shape))
                return self._temporal.update(detections)
            if self._missing_frames > self._cfg.max_missing_frames:
                self._last_board_bbox = None
                self._last_localization = None
                self._last_coarse_bbox = None
                self._last_coarse_bboxes = []
                self._temporal.reset()
                LOGGER.debug("temporal state reset after %d missing board frames", self._missing_frames)
                return []
            return self._temporal.update([])

        if self._cfg.enable_tracking:
            self._last_board_bbox = localization.bbox
            self._last_localization = localization
        else:
            self._last_board_bbox = None
            self._last_localization = None
        self._missing_frames = 0

        detections = [Detection(label="BOARD", score=localization.score, bbox=localization.bbox)]
        detections.extend(self._detect_components(localization, frame.shape))
        return self._temporal.update(detections)

    def _reuse_last_localization(self, frame: np.ndarray) -> BoardLocalization | None:
        if not self._cfg.enable_tracking or self._last_localization is None:
            return None
        if self._missing_frames > self._cfg.max_pose_reuse_frames:
            return None
        last = self._last_localization
        if last.warp_quality_score < self._cfg.reuse_min_warp_quality:
            return None

        out_h, out_w = last.warped.shape[:2]
        warped = cv.warpPerspective(frame, last.homography, (out_w, out_h))
        decay = float(np.clip(self._cfg.reuse_score_decay, 0.50, 1.0)) ** max(1, self._missing_frames)
        return BoardLocalization(
            quad=last.quad,
            bbox=last.bbox,
            homography=last.homography,
            h_inv=last.h_inv,
            warped=warped,
            score=last.score * decay,
            warp_quality_score=last.warp_quality_score * decay,
            geometry_score=last.geometry_score,
            verify_score=last.verify_score,
            objectness_score=last.objectness_score,
            pcb_structure_score=last.pcb_structure_score,
            tightness_score=last.tightness_score,
        )

    def _detect_components(self, localization: BoardLocalization, frame_shape: tuple[int, ...]) -> list[Detection]:
        out: list[Detection] = []
        for spec in self._component_specs:
            matcher = self._component_matchers.get(spec.label)
            if matcher is None:
                continue
            if localization.warp_quality_score < spec.min_warp_quality_score:
                self._log_component_rejection(
                    spec,
                    "low_warp_quality",
                    localization,
                    None,
                )
                continue

            roi_bbox = self._roi_to_bbox(spec.roi, localization.warped.shape)
            crop = localization.warped[roi_bbox.y1:roi_bbox.y2, roi_bbox.x1:roi_bbox.x2]
            if crop.size == 0:
                self._log_component_rejection(spec, "empty_roi", localization, None)
                continue

            visibility_score = self._roi_visibility_score(spec.label, crop)
            match_result = self._detect_component_in_roi(matcher, crop)
            detection = self._select_detection_with_visibility(
                spec,
                match_result,
                visibility_score,
                localization.warp_quality_score,
            )
            if detection is None or detection.score < spec.score_threshold:
                self._log_component_rejection(
                    spec,
                    match_result.reason or "low_template_score",
                    localization,
                    match_result,
                    visibility_score,
                )
                fallback = self._layout_fallback_detection(spec, localization, frame_shape, match_result, visibility_score)
                if fallback is not None:
                    out.append(fallback)
                continue
            if visibility_score < spec.min_visibility_score:
                self._log_component_rejection(
                    spec,
                    "low_roi_visibility",
                    localization,
                    match_result,
                    visibility_score,
                )
                fallback = self._layout_fallback_detection(spec, localization, frame_shape, match_result, visibility_score)
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
                self._log_component_rejection(spec, "homography_mapping_failed", localization, match_result, visibility_score)
                fallback = self._layout_fallback_detection(spec, localization, frame_shape, match_result, visibility_score)
                if fallback is not None:
                    out.append(fallback)
                continue
            sanity_reason = self._component_sanity_reason(spec, mapped_bbox, localization.bbox)
            if sanity_reason is not None:
                self._log_component_rejection(spec, sanity_reason, localization, match_result, visibility_score)
                fallback = self._layout_fallback_detection(spec, localization, frame_shape, match_result, visibility_score)
                if fallback is not None:
                    out.append(fallback)
                continue

            self._log_component_acceptance(spec, detection.score, localization, match_result, visibility_score)
            out.append(Detection(label=spec.label, score=detection.score, bbox=mapped_bbox))
        return out

    def _select_detection_with_visibility(
        self,
        spec: ComponentSpec,
        match_result: TemplateMatchResult,
        visibility_score: float,
        warp_quality_score: float,
    ) -> Detection | None:
        detection = match_result.detection
        if detection is not None:
            if spec.visibility_weight <= 0.0:
                return detection
            fused_score = self._fused_component_score(
                match_result.best_score,
                visibility_score,
                spec.visibility_weight,
                warp_quality_score,
                spec.warp_quality_weight,
            )
            return Detection(label=detection.label, score=fused_score, bbox=detection.bbox)

        if match_result.reason != "low_score" or match_result.candidate is None:
            return None
        if visibility_score < spec.min_visibility_score:
            return None
        fused_score = self._fused_component_score(
            match_result.best_score,
            visibility_score,
            spec.visibility_weight,
            warp_quality_score,
            spec.warp_quality_weight,
        )
        if fused_score < spec.score_threshold:
            return None
        return Detection(label=match_result.candidate.label, score=fused_score, bbox=match_result.candidate.bbox)

    @staticmethod
    def _fused_component_score(
        template_score: float,
        visibility_score: float,
        visibility_weight: float,
        warp_quality_score: float = 0.0,
        warp_quality_weight: float = 0.0,
    ) -> float:
        weight = float(np.clip(visibility_weight, 0.0, 0.45))
        score = (1.0 - weight) * template_score + weight * visibility_score
        warp_support = np.clip((warp_quality_score - 0.58) / 0.32, 0.0, 1.0) * np.clip(visibility_score, 0.0, 1.0)
        score += float(np.clip(warp_quality_weight, 0.0, 0.08)) * warp_support
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _detect_component_in_roi(matcher: TemplateMatcher, crop: np.ndarray) -> TemplateMatchResult:
        if hasattr(matcher, "detect_best_with_stats"):
            return matcher.detect_best_with_stats(crop)
        detection = matcher.detect_best(crop)
        best_score = detection.score if detection is not None else -1.0
        reason = "" if detection is not None else "low_template_score"
        return TemplateMatchResult(detection, best_score, -1.0, 1.0, reason)

    def _layout_fallback_detection(
        self,
        spec: ComponentSpec,
        localization: BoardLocalization,
        frame_shape: tuple[int, ...],
        match_result: TemplateMatchResult | None = None,
        visibility_score: float = -1.0,
    ) -> Detection | None:
        if spec.layout_roi is None or spec.layout_fallback_score <= 0.0:
            return None
        if localization.score < spec.layout_fallback_min_board_score:
            self._log_layout_rejection(spec, "low_board_score", localization, match_result, visibility_score)
            return None
        if localization.warp_quality_score < spec.layout_fallback_min_warp_quality:
            self._log_layout_rejection(spec, "low_warp_quality", localization, match_result, visibility_score)
            return None
        best_match_score = match_result.best_score if match_result is not None else -1.0
        if visibility_score < spec.layout_fallback_min_visibility_score:
            self._log_layout_rejection(spec, "low_layout_visibility", localization, match_result, visibility_score)
            return None
        fallback_evidence = self._fused_component_score(
            best_match_score,
            max(0.0, visibility_score),
            max(spec.visibility_weight, 0.35),
            localization.warp_quality_score,
            spec.warp_quality_weight,
        )
        if spec.layout_fallback_min_match_score > 0.0 and fallback_evidence < spec.layout_fallback_min_match_score:
            self._log_layout_rejection(spec, "low_layout_match_score", localization, match_result, visibility_score)
            return None

        canonical_bbox = self._roi_to_bbox(spec.layout_roi, localization.warped.shape)
        mapped_bbox = map_bbox_with_homography(canonical_bbox, localization.h_inv, frame_shape)
        if mapped_bbox is None or mapped_bbox.area() <= 0:
            self._log_layout_rejection(spec, "homography_mapping_failed", localization, match_result)
            return None
        sanity_reason = self._component_sanity_reason(spec, mapped_bbox, localization.bbox)
        if sanity_reason is not None:
            self._log_layout_rejection(spec, sanity_reason, localization, match_result)
            return None

        score = min(float(localization.score), float(spec.layout_fallback_score))
        LOGGER.debug(
            "layout fallback accepted: label=%s score=%.3f board_score=%.3f warp_quality=%.3f "
            "match=%.3f visibility=%.3f evidence=%.3f",
            spec.label,
            score,
            localization.score,
            localization.warp_quality_score,
            best_match_score,
            visibility_score,
            fallback_evidence,
        )
        return Detection(label=spec.label, score=score, bbox=mapped_bbox)

    @staticmethod
    def _roi_visibility_score(label: str, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = crop if crop.ndim == 2 else np.mean(crop, axis=2).astype(np.uint8)
        contrast = float(np.std(gray))
        contrast_score = float(np.clip((contrast - 14.0) / 48.0, 0.0, 1.0))
        gy, gx = np.gradient(gray.astype(np.float32))
        gradient = np.sqrt(gx * gx + gy * gy)
        edge_score = float(np.clip((float(np.mean(gradient > 18.0)) - 0.025) / 0.18, 0.0, 1.0))

        hsv = None
        if crop.ndim == 3:
            hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
        if label == "ESP32":
            dark_ratio = float(np.mean(gray < 130))
            dark_score = float(np.clip((dark_ratio - 0.22) / 0.45, 0.0, 1.0))
            shape_score = BoardFirstDetector._roi_rect_shape_score(gray < 145, 0.12, 0.82, 0.70, 2.80)
            return float(
                np.clip(
                    0.32 * contrast_score + 0.28 * edge_score + 0.22 * dark_score + 0.18 * shape_score,
                    0.0,
                    1.0,
                )
            )
        if label == "USB_PORT" and hsv is not None:
            bright_low_sat = float(np.mean((hsv[:, :, 2] > 135) & (hsv[:, :, 1] < 115)))
            metal_score = float(np.clip((bright_low_sat - 0.035) / 0.22, 0.0, 1.0))
            shape_score = BoardFirstDetector._roi_rect_shape_score(
                (hsv[:, :, 2] > 125) & (hsv[:, :, 1] < 135),
                0.025,
                0.55,
                0.65,
                2.60,
            )
            return float(
                np.clip(
                    0.30 * contrast_score + 0.32 * edge_score + 0.22 * metal_score + 0.16 * shape_score,
                    0.0,
                    1.0,
                )
            )
        if label == "JST_CONNECTOR" and hsv is not None:
            bright_plastic = float(np.mean((hsv[:, :, 2] > 145) & (hsv[:, :, 1] < 150)))
            plastic_score = float(np.clip((bright_plastic - 0.045) / 0.26, 0.0, 1.0))
            shape_score = BoardFirstDetector._roi_rect_shape_score(
                (hsv[:, :, 2] > 138) & (hsv[:, :, 1] < 170),
                0.030,
                0.62,
                0.65,
                2.40,
            )
            return float(
                np.clip(
                    0.24 * contrast_score + 0.25 * edge_score + 0.35 * plastic_score + 0.16 * shape_score,
                    0.0,
                    1.0,
                )
            )
        if label == "RESET_BUTTON":
            local_edges = float(np.mean(gradient > 22.0))
            small_edge_score = float(np.clip((local_edges - 0.035) / 0.16, 0.0, 1.0))
            bright_shape = BoardFirstDetector._roi_rect_shape_score(gray > 135, 0.006, 0.28, 0.65, 2.20)
            dark_shape = BoardFirstDetector._roi_rect_shape_score(gray < 115, 0.006, 0.28, 0.65, 2.20)
            shape_score = max(bright_shape, dark_shape)
            return float(np.clip(0.34 * contrast_score + 0.36 * small_edge_score + 0.30 * shape_score, 0.0, 1.0))
        return float(np.clip(0.55 * contrast_score + 0.45 * edge_score, 0.0, 1.0))

    @staticmethod
    def _roi_rect_shape_score(
        mask: np.ndarray,
        min_area_ratio: float,
        max_area_ratio: float,
        min_aspect: float,
        max_aspect: float,
    ) -> float:
        if mask.size == 0:
            return 0.0
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, mask.shape[0] * mask.shape[1]))
        best = 0.0
        for contour in contours:
            area_ratio = float(cv.contourArea(contour)) / roi_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            x, y, w, h = cv.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            box_area = float(w * h)
            rectangularity = float(np.clip(cv.contourArea(contour) / max(1.0, box_area), 0.0, 1.0))
            aspect = w / max(1.0, float(h))
            normalized_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
            if normalized_aspect < min_aspect or normalized_aspect > max_aspect:
                continue
            target_mid = 0.5 * (min_area_ratio + max_area_ratio)
            area_score = float(np.exp(-abs(np.log(max(1e-6, area_ratio) / max(1e-6, target_mid)))))
            aspect_mid = 0.5 * (min_aspect + max_aspect)
            aspect_score = float(np.exp(-0.8 * abs(np.log(max(1e-6, normalized_aspect) / max(1e-6, aspect_mid)))))
            best = max(best, 0.45 * rectangularity + 0.35 * area_score + 0.20 * aspect_score)
        return float(np.clip(best, 0.0, 1.0))

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
        return BoardFirstDetector._component_sanity_reason(spec, component_bbox, board_bbox) is None

    @staticmethod
    def _component_sanity_reason(spec: ComponentSpec, component_bbox: BBox, board_bbox: BBox) -> str | None:
        board_area = max(1, board_bbox.area())
        component_area_ratio = component_bbox.area() / board_area
        if component_area_ratio < spec.min_board_area_ratio or component_area_ratio > spec.max_board_area_ratio:
            return "component_area_ratio"

        width = max(1, component_bbox.width())
        height = max(1, component_bbox.height())
        aspect = width / height
        normalized_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
        if (
            normalized_aspect < spec.min_normalized_aspect_ratio
            or normalized_aspect > spec.max_normalized_aspect_ratio
        ):
            return "component_aspect_ratio"

        overlap_ratio = BoardFirstDetector._overlap_ratio(component_bbox, board_bbox)
        if overlap_ratio < spec.min_board_overlap_ratio:
            return "component_board_overlap"

        return None

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
            LOGGER.debug(
                "coarse board hints refreshed: count=%d scores=%s",
                len(coarse_candidates),
                [round(candidate.score, 3) for candidate in coarse_candidates if candidate is not None],
            )
        return list(self._last_coarse_bboxes)

    @staticmethod
    def _log_component_rejection(
        spec: ComponentSpec,
        reason: str,
        localization: BoardLocalization,
        match_result: TemplateMatchResult | None,
        visibility_score: float = -1.0,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "component rejected: label=%s reason=%s board_score=%.3f warp_quality=%.3f match=%.3f "
            "second=%.3f margin=%.3f visibility=%.3f",
            spec.label,
            reason,
            localization.score,
            localization.warp_quality_score,
            match_result.best_score if match_result is not None else -1.0,
            match_result.second_score if match_result is not None else -1.0,
            match_result.score_margin if match_result is not None else -1.0,
            visibility_score,
        )

    @staticmethod
    def _log_component_acceptance(
        spec: ComponentSpec,
        fused_score: float,
        localization: BoardLocalization,
        match_result: TemplateMatchResult | None,
        visibility_score: float,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "component accepted: label=%s score=%.3f board_score=%.3f warp_quality=%.3f "
            "match=%.3f second=%.3f margin=%.3f visibility=%.3f visibility_weight=%.3f warp_weight=%.3f",
            spec.label,
            fused_score,
            localization.score,
            localization.warp_quality_score,
            match_result.best_score if match_result is not None else -1.0,
            match_result.second_score if match_result is not None else -1.0,
            match_result.score_margin if match_result is not None else -1.0,
            visibility_score,
            spec.visibility_weight,
            spec.warp_quality_weight,
        )

    @staticmethod
    def _log_layout_rejection(
        spec: ComponentSpec,
        reason: str,
        localization: BoardLocalization,
        match_result: TemplateMatchResult | None,
        visibility_score: float = -1.0,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "layout fallback rejected: label=%s reason=%s board_score=%.3f min_board=%.3f "
            "warp_quality=%.3f min_warp=%.3f match=%.3f min_match=%.3f visibility=%.3f min_visibility=%.3f",
            spec.label,
            reason,
            localization.score,
            spec.layout_fallback_min_board_score,
            localization.warp_quality_score,
            spec.layout_fallback_min_warp_quality,
            match_result.best_score if match_result is not None else -1.0,
            spec.layout_fallback_min_match_score,
            visibility_score,
            spec.layout_fallback_min_visibility_score,
        )
