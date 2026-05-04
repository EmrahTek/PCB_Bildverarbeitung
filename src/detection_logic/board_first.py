"""Board-first component detector.

This module localizes the PCB, warps it into canonical coordinates, searches
component ROIs with template matchers, and tracks boxes across live frames.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- logging: https://docs.python.org/3/library/logging.html
"""

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

TRACK_ACQUIRE = "ACQUIRE"
TRACK_LOCKED = "LOCKED"
TRACK_LOCAL_SEARCH = "LOCAL_SEARCH"
TRACK_LOST = "LOST"


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
    keep_score_threshold: float = 0.0
    keep_min_visibility_score: float = 0.0
    local_search_expansion: float = 0.55
    track_max_missing: int = 2
    track_smoothing_alpha: float = 0.55
    position_prior_weight: float = 0.0
    min_position_prior_acquire: float = 0.0
    min_position_prior_keep: float = 0.0
    persistence_decay: float = 0.88
    visibility_upscale: float = 1.0
    layout_anchor: bool = False
    output_bbox_pad_left: float = 0.0
    output_bbox_pad_right: float = 0.0
    output_bbox_pad_top: float = 0.0
    output_bbox_pad_bottom: float = 0.0


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
    board_smoothing_alpha: float = 0.55
    board_smoothing_min_quality: float = 0.58
    board_smoothing_max_shift: float = 0.055
    board_pose_max_area_growth: float = 0.22
    board_pose_max_quality_drop: float = 0.16
    board_pose_max_tightness_drop: float = 0.20
    board_pose_quality_margin: float = 0.05
    board_pose_reuse_decay: float = 0.98
    board_bbox_pad_left: float = 0.0
    board_bbox_pad_right: float = 0.0
    board_bbox_pad_top: float = 0.0
    board_bbox_pad_bottom: float = 0.0


@dataclass(frozen=True)
class _ComponentTrack:
    canonical_bbox: BBox
    score: float
    visibility_score: float
    missing_frames: int = 0
    hits: int = 1
    state: str = TRACK_ACQUIRE


@dataclass(frozen=True)
class _ComponentCandidate:
    canonical_bbox: BBox
    score: float
    visibility_score: float
    match_result: TemplateMatchResult
    mode: str


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
        self._component_tracks: dict[str, _ComponentTrack] = {}
        self._missing_frames = 0
        self._frame_index = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frame_index += 1

        coarse_hints = self._maybe_refresh_coarse_hints(frame)
        tracking_hint = self._last_board_bbox if self._cfg.enable_tracking else None

        localization = self._localize_from_hints(frame, coarse_hints, tracking_hint)
        if localization is None:
            localization = self._tracked_board_rescue(frame, tracking_hint)

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
                component_detections = self._detect_components(reused, frame.shape)
                return self._emit_detections(
                    reused,
                    component_detections,
                    board_source="reused",
                    frame_shape=frame.shape,
                )
            if self._missing_frames > self._cfg.max_missing_frames:
                self._last_board_bbox = None
                self._last_localization = None
                self._last_coarse_bbox = None
                self._last_coarse_bboxes = []
                self._component_tracks.clear()
                self._temporal.reset()
                LOGGER.debug("temporal state reset after %d missing board frames", self._missing_frames)
                return []
            self._temporal.update([])
            return []

        localization = self._stabilize_localization(frame, localization)
        if self._cfg.enable_tracking:
            self._last_board_bbox = localization.bbox
            self._last_localization = localization
        else:
            self._last_board_bbox = None
            self._last_localization = None
            self._component_tracks.clear()
        self._missing_frames = 0

        component_detections = self._detect_components(localization, frame.shape)
        return self._emit_detections(
            localization,
            component_detections,
            board_source="pose",
            frame_shape=frame.shape,
        )

    def _reuse_last_localization(self, frame: np.ndarray) -> BoardLocalization | None:
        if not self._cfg.enable_tracking or self._last_localization is None:
            return None
        if self._missing_frames > self._cfg.max_pose_reuse_frames:
            return None
        last = self._last_localization
        if last.warp_quality_score < self._cfg.reuse_min_warp_quality:
            return None

        decay = float(np.clip(self._cfg.reuse_score_decay, 0.50, 1.0)) ** max(1, self._missing_frames)
        return self._localization_from_previous_pose(frame, last, decay)

    def _localization_from_previous_pose(
        self,
        frame: np.ndarray,
        previous: BoardLocalization,
        decay: float,
    ) -> BoardLocalization:
        out_h, out_w = previous.warped.shape[:2]
        warped = cv.warpPerspective(frame, previous.homography, (out_w, out_h))
        return BoardLocalization(
            quad=previous.quad,
            bbox=previous.bbox,
            homography=previous.homography,
            h_inv=previous.h_inv,
            warped=warped,
            score=previous.score * decay,
            warp_quality_score=previous.warp_quality_score * decay,
            geometry_score=previous.geometry_score,
            verify_score=previous.verify_score,
            objectness_score=previous.objectness_score,
            pcb_structure_score=previous.pcb_structure_score,
            tightness_score=previous.tightness_score,
            canonical_structure_score=previous.canonical_structure_score,
            edge_grid_score=previous.edge_grid_score,
            board_identity_score=previous.board_identity_score,
            header_score=previous.header_score,
            corner_hole_score=previous.corner_hole_score,
            connector_score=previous.connector_score,
            skin_ratio=previous.skin_ratio,
        )

    def _stabilize_localization(self, frame: np.ndarray, localization: BoardLocalization) -> BoardLocalization:
        if not self._cfg.enable_tracking or self._last_localization is None:
            return localization
        last = self._last_localization
        if last.warp_quality_score < self._cfg.board_smoothing_min_quality:
            return localization

        reject_reason = self._pose_rejection_reason(localization, last)
        if reject_reason is not None:
            reused = self._localization_from_previous_pose(
                frame,
                last,
                float(np.clip(self._cfg.board_pose_reuse_decay, 0.70, 1.0)),
            )
            LOGGER.debug(
                "board pose rejected and previous pose reused: reason=%s current_bbox=%s previous_bbox=%s "
                "current_score=%.3f previous_score=%.3f current_warp=%.3f previous_warp=%.3f "
                "current_tightness=%.3f previous_tightness=%.3f",
                reject_reason,
                localization.bbox,
                last.bbox,
                localization.score,
                last.score,
                localization.warp_quality_score,
                last.warp_quality_score,
                localization.tightness_score,
                last.tightness_score,
            )
            return reused

        if localization.warp_quality_score < self._cfg.board_smoothing_min_quality:
            return localization

        shift = self._normalized_quad_shift(localization.quad, last.quad, last.bbox)
        if shift > self._cfg.board_smoothing_max_shift:
            LOGGER.debug(
                "board pose smoothing skipped: shift=%.3f max=%.3f score=%.3f warp_quality=%.3f",
                shift,
                self._cfg.board_smoothing_max_shift,
                localization.score,
                localization.warp_quality_score,
            )
            return localization

        alpha = float(np.clip(self._cfg.board_smoothing_alpha, 0.05, 1.0))
        smoothed_quad = (alpha * localization.quad + (1.0 - alpha) * last.quad).astype(np.float32)
        out_h, out_w = localization.warped.shape[:2]
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        homography = cv.getPerspectiveTransform(smoothed_quad, dst)
        try:
            h_inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return localization
        warped = cv.warpPerspective(frame, homography, (out_w, out_h))
        bbox = self._quad_to_bbox(smoothed_quad, frame.shape)
        LOGGER.debug(
            "board pose smoothed: alpha=%.2f shift=%.3f bbox=%s warp_quality=%.3f",
            alpha,
            shift,
            bbox,
            localization.warp_quality_score,
        )
        return BoardLocalization(
            quad=smoothed_quad,
            bbox=bbox,
            homography=homography,
            h_inv=h_inv,
            warped=warped,
            score=localization.score,
            warp_quality_score=localization.warp_quality_score,
            geometry_score=localization.geometry_score,
            verify_score=localization.verify_score,
            objectness_score=localization.objectness_score,
            pcb_structure_score=localization.pcb_structure_score,
            tightness_score=localization.tightness_score,
            canonical_structure_score=localization.canonical_structure_score,
            edge_grid_score=localization.edge_grid_score,
            board_identity_score=localization.board_identity_score,
            header_score=localization.header_score,
            corner_hole_score=localization.corner_hole_score,
            connector_score=localization.connector_score,
            skin_ratio=localization.skin_ratio,
        )

    def _emit_detections(
        self,
        localization: BoardLocalization,
        component_detections: list[Detection],
        *,
        board_source: str,
        frame_shape: tuple[int, ...],
    ) -> list[Detection]:
        stable_components = self._temporal.update(component_detections)
        stable_components.sort(key=lambda det: det.label)
        board_bbox = self._display_board_bbox(localization.bbox, frame_shape)
        board_detection = Detection(label="BOARD", score=localization.score, bbox=board_bbox)
        LOGGER.debug(
            "board bbox generated: source=%s bbox=%s raw_bbox=%s score=%.3f warp_quality=%.3f "
            "geometry=%.3f verify=%.3f objectness=%.3f structure=%.3f canonical=%.3f "
            "identity=%.3f header=%.3f corner=%.3f connector=%.3f edge_grid=%.3f "
            "tightness=%.3f skin=%.3f quad_area=%.1f components_raw=%d components_stable=%d",
            board_source,
            board_bbox,
            localization.bbox,
            localization.score,
            localization.warp_quality_score,
            localization.geometry_score,
            localization.verify_score,
            localization.objectness_score,
            localization.pcb_structure_score,
            localization.canonical_structure_score,
            localization.board_identity_score,
            localization.header_score,
            localization.corner_hole_score,
            localization.connector_score,
            localization.edge_grid_score,
            localization.tightness_score,
            localization.skin_ratio,
            self._quad_area(localization.quad),
            len(component_detections),
            len(stable_components),
        )
        return [board_detection, *stable_components]

    def _display_board_bbox(self, bbox: BBox, frame_shape: tuple[int, ...]) -> BBox:
        return self._pad_bbox_for_output(
            bbox,
            frame_shape,
            left=self._cfg.board_bbox_pad_left,
            right=self._cfg.board_bbox_pad_right,
            top=self._cfg.board_bbox_pad_top,
            bottom=self._cfg.board_bbox_pad_bottom,
        )

    @staticmethod
    def _pad_bbox_for_output(
        bbox: BBox,
        frame_shape: tuple[int, ...],
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> BBox:
        pad_left = max(0.0, left)
        pad_right = max(0.0, right)
        pad_top = max(0.0, top)
        pad_bottom = max(0.0, bottom)
        if pad_left <= 0.0 and pad_right <= 0.0 and pad_top <= 0.0 and pad_bottom <= 0.0:
            return bbox

        frame_h, frame_w = frame_shape[:2]
        width = max(1, bbox.width())
        height = max(1, bbox.height())
        return BBox(
            int(max(0, np.floor(bbox.x1 - width * pad_left))),
            int(max(0, np.floor(bbox.y1 - height * pad_top))),
            int(min(frame_w - 1, np.ceil(bbox.x2 + width * pad_right))),
            int(min(frame_h - 1, np.ceil(bbox.y2 + height * pad_bottom))),
        )

    def _output_component_bbox(self, spec: ComponentSpec, bbox: BBox, frame_shape: tuple[int, ...]) -> BBox:
        return self._pad_bbox_for_output(
            bbox,
            frame_shape,
            left=spec.output_bbox_pad_left,
            right=spec.output_bbox_pad_right,
            top=spec.output_bbox_pad_top,
            bottom=spec.output_bbox_pad_bottom,
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
                self._mark_component_missing(spec)
                continue

            roi_bbox = self._roi_to_bbox(spec.roi, localization.warped.shape)
            if roi_bbox.area() <= 0:
                self._log_component_rejection(spec, "empty_roi", localization, None)
                self._mark_component_missing(spec)
                continue

            track = self._component_tracks.get(spec.label) if self._cfg.enable_tracking else None
            active_track = self._track_is_active(spec, track)
            candidate = self._detect_component_with_lock(
                spec,
                matcher,
                localization,
                roi_bbox,
                track,
            )

            if candidate is None:
                if active_track:
                    LOGGER.debug(
                        "component full search deferred: label=%s state=%s missing=%d previous=%s",
                        spec.label,
                        track.state if track is not None else TRACK_LOST,
                        track.missing_frames if track is not None else -1,
                        track.canonical_bbox if track is not None else None,
                    )
                    self._mark_component_missing(spec)
                    continue
                fallback_crop = localization.warped[roi_bbox.y1:roi_bbox.y2, roi_bbox.x1:roi_bbox.x2]
                visibility_score = self._roi_visibility_score(spec.label, fallback_crop) if fallback_crop.size else -1.0
                match_result = self._detect_component_in_roi(matcher, fallback_crop) if fallback_crop.size else None
                fallback = self._layout_fallback_detection(spec, localization, frame_shape, match_result, visibility_score)
                if fallback is not None:
                    canonical_fallback = self._roi_to_bbox(spec.layout_roi, localization.warped.shape) if spec.layout_roi else None
                    if canonical_fallback is not None:
                        self._update_component_track(spec, canonical_fallback, fallback.score, max(0.0, visibility_score), "layout")
                    out.append(fallback)
                    continue
                self._mark_component_missing(spec)
                continue

            mapped_bbox = map_bbox_with_homography(candidate.canonical_bbox, localization.h_inv, frame_shape)
            if mapped_bbox is None or mapped_bbox.area() <= 0:
                self._log_component_rejection(
                    spec,
                    "homography_mapping_failed",
                    localization,
                    candidate.match_result,
                    candidate.visibility_score,
                )
                if active_track:
                    self._mark_component_missing(spec)
                    continue
                fallback = self._layout_fallback_detection(
                    spec,
                    localization,
                    frame_shape,
                    candidate.match_result,
                    candidate.visibility_score,
                )
                if fallback is not None:
                    out.append(fallback)
                self._mark_component_missing(spec)
                continue
            sanity_reason = self._component_sanity_reason(spec, mapped_bbox, localization.bbox)
            if sanity_reason is not None:
                self._log_component_rejection(
                    spec,
                    sanity_reason,
                    localization,
                    candidate.match_result,
                    candidate.visibility_score,
                )
                if active_track:
                    self._mark_component_missing(spec)
                    continue
                fallback = self._layout_fallback_detection(
                    spec,
                    localization,
                    frame_shape,
                    candidate.match_result,
                    candidate.visibility_score,
                )
                if fallback is not None:
                    out.append(fallback)
                self._mark_component_missing(spec)
                continue

            self._log_component_acceptance(
                spec,
                candidate.score,
                localization,
                candidate.match_result,
                candidate.visibility_score,
                candidate.mode,
            )
            self._update_component_track(
                spec,
                candidate.canonical_bbox,
                candidate.score,
                candidate.visibility_score,
                candidate.mode,
            )
            mapped_bbox = self._output_component_bbox(spec, mapped_bbox, frame_shape)
            out.append(Detection(label=spec.label, score=candidate.score, bbox=mapped_bbox))
        return out

    def _detect_component_with_lock(
        self,
        spec: ComponentSpec,
        matcher: TemplateMatcher,
        localization: BoardLocalization,
        roi_bbox: BBox,
        track: _ComponentTrack | None,
    ) -> _ComponentCandidate | None:
        if self._track_is_active(spec, track):
            locked_window = self._locked_search_window(track.canonical_bbox, roi_bbox, localization.warped.shape, spec)
            search_state = TRACK_LOCAL_SEARCH if track.missing_frames > 0 or track.state == TRACK_LOCAL_SEARCH else TRACK_LOCKED
            mode = "local_search" if search_state == TRACK_LOCAL_SEARCH else "locked"
            LOGGER.debug(
                "component local search: label=%s state=%s window=%s previous=%s missing=%d",
                spec.label,
                search_state,
                locked_window,
                track.canonical_bbox,
                track.missing_frames,
            )
            keep_candidate = self._match_component_window(
                spec,
                matcher,
                localization,
                locked_window,
                threshold=self._keep_score_threshold(spec),
                min_visibility=self._keep_min_visibility_score(spec),
                mode=mode,
                track=track,
            )
            if keep_candidate is not None:
                return keep_candidate

            persisted = self._persistent_component_candidate(spec, localization, locked_window, track)
            if persisted is not None:
                return persisted

            LOGGER.debug(
                "component local search miss: label=%s state=%s missing=%d full_search=deferred",
                spec.label,
                search_state,
                track.missing_frames,
            )
            return None

        return self._match_component_window(
            spec,
            matcher,
            localization,
            roi_bbox,
            threshold=spec.score_threshold,
            min_visibility=spec.min_visibility_score,
            mode="full",
            track=track,
        )

    def _match_component_window(
        self,
        spec: ComponentSpec,
        matcher: TemplateMatcher,
        localization: BoardLocalization,
        search_bbox: BBox,
        *,
        threshold: float,
        min_visibility: float,
        mode: str,
        track: _ComponentTrack | None,
    ) -> _ComponentCandidate | None:
        search_bbox = self._clip_bbox(search_bbox, localization.warped.shape)
        crop = localization.warped[search_bbox.y1:search_bbox.y2, search_bbox.x1:search_bbox.x2]
        if crop.size == 0:
            return None

        window_visibility_score = self._roi_visibility_score(spec.label, crop)
        match_result = self._detect_component_in_roi(matcher, crop)
        detection = match_result.detection
        layout_anchor_requested = self._should_use_layout_anchor(spec, localization, match_result)
        layout_anchor_used = False
        low_score_candidate_used = False
        if detection is None and match_result.reason == "low_score":
            detection = match_result.candidate
            low_score_candidate_used = detection is not None
        if detection is None and layout_anchor_requested and spec.layout_roi is not None:
            canonical_bbox = self._roi_to_bbox(spec.layout_roi, localization.warped.shape)
            layout_anchor_used = True
        elif detection is None:
            if spec.label in {"USB_PORT", "JST_CONNECTOR"}:
                coverage = self._right_connector_coverage_score(localization.warped)
                LOGGER.debug(
                    "component evidence missing: label=%s mode=%s state=%s connector_coverage=%.3f "
                    "board_score=%.3f warp_quality=%.3f window=%s",
                    spec.label,
                    mode,
                    "right_edge_undercovered" if coverage < 0.22 else "no_component_evidence",
                    coverage,
                    localization.score,
                    localization.warp_quality_score,
                    search_bbox,
                )
            LOGGER.debug(
                "component window rejected: label=%s mode=%s reason=%s threshold=%.3f "
                "match=%.3f visibility=%.3f min_visibility=%.3f window=%s",
                spec.label,
                mode,
                match_result.reason or "low_score",
                threshold,
                match_result.best_score,
                window_visibility_score,
                min_visibility,
                search_bbox,
            )
            return None
        else:
            canonical_bbox = BBox(
                search_bbox.x1 + detection.bbox.x1,
                search_bbox.y1 + detection.bbox.y1,
                search_bbox.x1 + detection.bbox.x2,
                search_bbox.y1 + detection.bbox.y2,
            )
            if layout_anchor_requested or self._should_snap_connector_to_layout(spec, match_result, low_score_candidate_used):
                canonical_bbox = self._roi_to_bbox(spec.layout_roi, localization.warped.shape)
                layout_anchor_used = layout_anchor_requested
        local_visibility_score = self._candidate_visibility_score(spec, localization.warped, canonical_bbox)
        visibility_score = self._combine_visibility_score(spec.label, window_visibility_score, local_visibility_score)
        if visibility_score < min_visibility:
            if spec.label in {"USB_PORT", "JST_CONNECTOR"}:
                coverage = self._right_connector_coverage_score(localization.warped)
                LOGGER.debug(
                    "component evidence weak: label=%s mode=%s state=%s connector_coverage=%.3f "
                    "visibility=%.3f min_visibility=%.3f board_score=%.3f warp_quality=%.3f",
                    spec.label,
                    mode,
                    "right_edge_undercovered" if coverage < 0.22 else "low_local_evidence",
                    coverage,
                    visibility_score,
                    min_visibility,
                    localization.score,
                    localization.warp_quality_score,
                )
            LOGGER.debug(
                "component window rejected: label=%s mode=%s reason=low_local_visibility "
                "visibility=%.3f local=%.3f window_visibility=%.3f min_visibility=%.3f window=%s canonical=%s",
                spec.label,
                mode,
                visibility_score,
                local_visibility_score,
                window_visibility_score,
                min_visibility,
                search_bbox,
                canonical_bbox,
            )
            return None

        layout_prior, track_prior, position_prior = self._position_priors(
            spec,
            canonical_bbox,
            localization.warped.shape,
            track,
        )
        min_position_prior = self._min_position_prior(spec, mode, track)
        if position_prior < min_position_prior:
            LOGGER.debug(
                "component window rejected: label=%s mode=%s reason=low_position_prior "
                "prior=%.3f min_prior=%.3f layout=%.3f track=%.3f window=%s canonical=%s",
                spec.label,
                mode,
                position_prior,
                min_position_prior,
                layout_prior,
                track_prior,
                search_bbox,
                canonical_bbox,
            )
            return None

        raw_score = match_result.best_score if match_result.best_score >= 0.0 else (detection.score if detection else 0.0)
        fused_score = self._fused_component_score(
            raw_score,
            visibility_score,
            spec.visibility_weight,
            localization.warp_quality_score,
            spec.warp_quality_weight,
        )
        score = self._score_with_position_prior(
            spec,
            canonical_bbox,
            fused_score,
            localization.warped.shape,
            track,
            priors=(layout_prior, track_prior, position_prior),
        )
        if layout_anchor_used and spec.layout_fallback_score > 0.0:
            anchored_score = min(float(localization.score), float(spec.layout_fallback_score))
            if anchored_score > score:
                LOGGER.debug(
                    "component layout anchor boosted: label=%s mode=%s score=%.3f anchored=%.3f "
                    "raw=%.3f visibility=%.3f board_score=%.3f warp_quality=%.3f canonical=%s",
                    spec.label,
                    mode,
                    score,
                    anchored_score,
                    raw_score,
                    visibility_score,
                    localization.score,
                    localization.warp_quality_score,
                    canonical_bbox,
                )
                score = anchored_score
        if score < threshold:
            LOGGER.debug(
                "component window rejected: label=%s mode=%s reason=low_prior_fused_score "
                "score=%.3f threshold=%.3f raw=%.3f window=%s canonical=%s",
                spec.label,
                mode,
                score,
                threshold,
                raw_score,
                search_bbox,
                canonical_bbox,
            )
            return None

        LOGGER.debug(
            "component window accepted: label=%s mode=%s score=%.3f raw=%.3f visibility=%.3f "
            "local_visibility=%.3f window_visibility=%.3f position_prior=%.3f match=%.3f "
            "threshold=%.3f window=%s canonical=%s",
            spec.label,
            mode,
            score,
            raw_score,
            visibility_score,
            local_visibility_score,
            window_visibility_score,
            position_prior,
            match_result.best_score,
            threshold,
            search_bbox,
            canonical_bbox,
        )
        candidate_mode = "layout_anchor" if layout_anchor_used else mode
        return _ComponentCandidate(canonical_bbox, score, visibility_score, match_result, candidate_mode)

    @staticmethod
    def _should_use_layout_anchor(
        spec: ComponentSpec,
        localization: BoardLocalization,
        match_result: TemplateMatchResult,
    ) -> bool:
        if not spec.layout_anchor or spec.layout_roi is None:
            return False
        if spec.layout_fallback_score <= 0.0:
            return False
        if localization.score < spec.layout_fallback_min_board_score:
            return False
        if localization.warp_quality_score < spec.layout_fallback_min_warp_quality:
            return False
        if match_result.best_score < 0.0:
            return spec.layout_fallback_min_match_score <= 0.0
        return match_result.best_score >= max(0.0, spec.layout_fallback_min_match_score)

    @staticmethod
    def _should_snap_connector_to_layout(
        spec: ComponentSpec,
        match_result: TemplateMatchResult,
        low_score_candidate_used: bool,
    ) -> bool:
        if spec.layout_roi is None or spec.label not in {"USB_PORT", "JST_CONNECTOR"}:
            return False
        if low_score_candidate_used:
            return True
        if match_result.best_score < 0.0:
            return False
        weak_match_threshold = max(0.14, spec.layout_fallback_min_match_score + 0.04)
        return match_result.best_score < weak_match_threshold

    def _persistent_component_candidate(
        self,
        spec: ComponentSpec,
        localization: BoardLocalization,
        search_bbox: BBox,
        track: _ComponentTrack,
    ) -> _ComponentCandidate | None:
        if track.missing_frames >= spec.track_max_missing:
            return None
        if localization.warp_quality_score < max(spec.min_warp_quality_score, spec.layout_fallback_min_warp_quality):
            return None

        crop = localization.warped[search_bbox.y1:search_bbox.y2, search_bbox.x1:search_bbox.x2]
        window_visibility_score = self._roi_visibility_score(spec.label, crop) if crop.size else 0.0
        local_visibility_score = self._candidate_visibility_score(spec, localization.warped, track.canonical_bbox)
        visibility_score = self._combine_visibility_score(spec.label, window_visibility_score, local_visibility_score)
        min_visibility = self._keep_min_visibility_score(spec)
        if visibility_score < min_visibility:
            LOGGER.debug(
                "component persistence rejected: label=%s reason=low_visibility visibility=%.3f "
                "local=%.3f window=%.3f min_visibility=%.3f",
                spec.label,
                visibility_score,
                local_visibility_score,
                window_visibility_score,
                min_visibility,
            )
            return None

        layout_prior, track_prior, position_prior = self._position_priors(
            spec,
            track.canonical_bbox,
            localization.warped.shape,
            track,
        )
        min_position_prior = self._min_position_prior(spec, "persistent", track)
        if position_prior < min_position_prior:
            LOGGER.debug(
                "component persistence rejected: label=%s reason=low_position_prior prior=%.3f "
                "min_prior=%.3f layout=%.3f track=%.3f",
                spec.label,
                position_prior,
                min_position_prior,
                layout_prior,
                track_prior,
            )
            return None

        score = float(np.clip(track.score * spec.persistence_decay, 0.0, 1.0))
        if score < self._keep_score_threshold(spec):
            LOGGER.debug(
                "component persistence rejected: label=%s reason=low_decayed_score score=%.3f threshold=%.3f "
                "previous=%.3f decay=%.3f",
                spec.label,
                score,
                self._keep_score_threshold(spec),
                track.score,
                spec.persistence_decay,
            )
            return None

        LOGGER.debug(
            "component kept by persistence: label=%s score=%.3f previous=%.3f decay=%.3f "
            "visibility=%.3f local_visibility=%.3f window_visibility=%.3f position_prior=%.3f "
            "missing=%d canonical=%s",
            spec.label,
            score,
            track.score,
            spec.persistence_decay,
            visibility_score,
            local_visibility_score,
            window_visibility_score,
            position_prior,
            track.missing_frames,
            track.canonical_bbox,
        )
        return _ComponentCandidate(
            canonical_bbox=track.canonical_bbox,
            score=score,
            visibility_score=visibility_score,
            match_result=TemplateMatchResult(None, -1.0, -1.0, 1.0, "persistent_track"),
            mode="persistent",
        )

    def _select_detection_with_visibility(
        self,
        spec: ComponentSpec,
        match_result: TemplateMatchResult,
        visibility_score: float,
        warp_quality_score: float,
        *,
        threshold: float | None = None,
        min_visibility: float | None = None,
    ) -> Detection | None:
        accept_threshold = spec.score_threshold if threshold is None else threshold
        visibility_threshold = spec.min_visibility_score if min_visibility is None else min_visibility
        if visibility_score < visibility_threshold:
            return None
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
        fused_score = self._fused_component_score(
            match_result.best_score,
            visibility_score,
            spec.visibility_weight,
            warp_quality_score,
            spec.warp_quality_weight,
        )
        if fused_score < accept_threshold:
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

    def _score_with_position_prior(
        self,
        spec: ComponentSpec,
        canonical_bbox: BBox,
        score: float,
        warped_shape: tuple[int, ...],
        track: _ComponentTrack | None,
        priors: tuple[float, float, float] | None = None,
    ) -> float:
        weight = float(np.clip(spec.position_prior_weight, 0.0, 0.35))
        if weight <= 0.0:
            return score
        if priors is None:
            layout_prior, track_prior, prior = self._position_priors(spec, canonical_bbox, warped_shape, track)
        else:
            layout_prior, track_prior, prior = priors
        fused = (1.0 - weight) * score + weight * prior
        LOGGER.debug(
            "component position prior: label=%s raw=%.3f prior=%.3f layout=%.3f track=%.3f fused=%.3f",
            spec.label,
            score,
            prior,
            layout_prior,
            track_prior,
            fused,
        )
        return float(np.clip(fused, 0.0, 1.0))

    def _position_priors(
        self,
        spec: ComponentSpec,
        canonical_bbox: BBox,
        warped_shape: tuple[int, ...],
        track: _ComponentTrack | None,
    ) -> tuple[float, float, float]:
        layout_prior = self._layout_position_prior(spec, canonical_bbox, warped_shape)
        track_prior = self._track_position_prior(canonical_bbox, track) if track is not None else 0.0
        return layout_prior, track_prior, max(layout_prior, track_prior)

    def _layout_position_prior(self, spec: ComponentSpec, canonical_bbox: BBox, warped_shape: tuple[int, ...]) -> float:
        expected_roi = spec.layout_roi or spec.roi
        expected_bbox = self._roi_to_bbox(expected_roi, warped_shape)
        scale_by_label = {
            "USB_PORT": 0.46,
            "JST_CONNECTOR": 0.44,
            "RESET_BUTTON": 0.46,
        }
        center_prior = self._center_prior(canonical_bbox, expected_bbox, scale=scale_by_label.get(spec.label, 0.65))
        if spec.label in {"USB_PORT", "JST_CONNECTOR"}:
            zone_prior = self._right_connector_zone_prior(spec.label, canonical_bbox, warped_shape)
            return float(np.clip(0.68 * center_prior + 0.32 * zone_prior, 0.0, 1.0))
        return center_prior

    @staticmethod
    def _right_connector_zone_prior(label: str, canonical_bbox: BBox, warped_shape: tuple[int, ...]) -> float:
        height, width = warped_shape[:2]
        cx, cy = BoardFirstDetector._bbox_center(canonical_bbox)
        x_frac = cx / max(1.0, float(width))
        y_frac = cy / max(1.0, float(height))

        if label == "USB_PORT":
            x_score = float(np.clip((x_frac - 0.60) / 0.30, 0.0, 1.0))
            y_score = float(np.exp(-abs(y_frac - 0.33) / 0.34))
        else:
            x_score = float(np.clip((x_frac - 0.58) / 0.32, 0.0, 1.0))
            y_score = float(np.exp(-abs(y_frac - 0.64) / 0.34))
        return float(np.clip(0.76 * x_score + 0.24 * y_score, 0.0, 1.0))

    @staticmethod
    def _track_position_prior(canonical_bbox: BBox, track: _ComponentTrack | None) -> float:
        if track is None:
            return 0.0
        return BoardFirstDetector._center_prior(canonical_bbox, track.canonical_bbox, scale=0.62)

    @staticmethod
    def _min_position_prior(spec: ComponentSpec, mode: str, track: _ComponentTrack | None) -> float:
        if track is not None and mode in {"locked", "local_search", "persistent"}:
            return float(np.clip(spec.min_position_prior_keep, 0.0, 1.0))
        return float(np.clip(spec.min_position_prior_acquire, 0.0, 1.0))

    @staticmethod
    def _center_prior(candidate: BBox, expected: BBox, *, scale: float) -> float:
        cx, cy = BoardFirstDetector._bbox_center(candidate)
        ex, ey = BoardFirstDetector._bbox_center(expected)
        diag = max(1.0, float(np.hypot(expected.width(), expected.height())))
        dist = float(np.hypot(cx - ex, cy - ey))
        return float(np.exp(-dist / max(1e-6, scale * diag)))

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
        canonical_bbox = self._roi_to_bbox(spec.layout_roi, localization.warped.shape)
        local_visibility_score = self._candidate_visibility_score(spec, localization.warped, canonical_bbox)
        evidence_visibility = self._combine_visibility_score(
            spec.label,
            max(0.0, visibility_score),
            local_visibility_score,
        )
        if evidence_visibility < spec.layout_fallback_min_visibility_score:
            self._log_layout_rejection(spec, "low_layout_visibility", localization, match_result, evidence_visibility)
            return None
        fallback_evidence = self._fused_component_score(
            best_match_score,
            max(0.0, evidence_visibility),
            max(spec.visibility_weight, 0.35),
            localization.warp_quality_score,
            spec.warp_quality_weight,
        )
        if spec.layout_fallback_min_match_score > 0.0 and fallback_evidence < spec.layout_fallback_min_match_score:
            self._log_layout_rejection(spec, "low_layout_match_score", localization, match_result, evidence_visibility)
            return None

        layout_prior, _track_prior, position_prior = self._position_priors(spec, canonical_bbox, localization.warped.shape, None)
        min_position_prior = float(np.clip(spec.min_position_prior_acquire, 0.0, 1.0))
        if position_prior < min_position_prior:
            self._log_layout_rejection(spec, "low_position_prior", localization, match_result, evidence_visibility)
            return None
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
            "match=%.3f visibility=%.3f local_visibility=%.3f position_prior=%.3f layout_prior=%.3f evidence=%.3f",
            spec.label,
            score,
            localization.score,
            localization.warp_quality_score,
            best_match_score,
            evidence_visibility,
            local_visibility_score,
            position_prior,
            layout_prior,
            fallback_evidence,
        )
        mapped_bbox = self._output_component_bbox(spec, mapped_bbox, frame_shape)
        return Detection(label=spec.label, score=score, bbox=mapped_bbox)

    def _candidate_visibility_score(self, spec: ComponentSpec, warped: np.ndarray, canonical_bbox: BBox) -> float:
        expansion_by_label = {
            "ESP32": 0.12,
            "USB_PORT": 0.22,
            "JST_CONNECTOR": 0.24,
            "RESET_BUTTON": 0.45,
        }
        expanded = self._expand_bbox(canonical_bbox, expansion_by_label.get(spec.label, 0.20), warped.shape)
        crop = warped[expanded.y1:expanded.y2, expanded.x1:expanded.x2]
        if crop.size == 0:
            return 0.0
        upscale = float(np.clip(spec.visibility_upscale, 1.0, 3.0))
        min_side = min(crop.shape[:2])
        if upscale > 1.0 and min_side < 120:
            crop = cv.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv.INTER_CUBIC)
        return self._roi_visibility_score(spec.label, crop)

    @staticmethod
    def _right_connector_coverage_score(warped: np.ndarray) -> float:
        if warped.size == 0:
            return 0.0
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV) if warped.ndim == 3 else None
        gray = warped if warped.ndim == 2 else np.mean(warped, axis=2).astype(np.uint8)
        h, w = gray.shape[:2]
        right_gray = gray[int(0.10 * h) : int(0.88 * h), int(0.62 * w) : int(0.99 * w)]
        if right_gray.size == 0:
            return 0.0
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        edges = cv.Canny(clahe.apply(right_gray), 45, 135)
        edge_score = float(np.clip((float(np.mean(edges > 0)) - 0.018) / 0.12, 0.0, 1.0))
        contrast_score = float(np.clip((float(np.std(right_gray)) - 10.0) / 42.0, 0.0, 1.0))
        bright_score = 0.0
        if hsv is not None:
            right_hsv = hsv[int(0.10 * h) : int(0.88 * h), int(0.62 * w) : int(0.99 * w)]
            bright_low_sat = float(np.mean((right_hsv[:, :, 2] > 140) & (right_hsv[:, :, 1] < 145)))
            bright_score = float(np.clip((bright_low_sat - 0.025) / 0.20, 0.0, 1.0))
        return float(np.clip(0.38 * edge_score + 0.32 * contrast_score + 0.30 * bright_score, 0.0, 1.0))

    @staticmethod
    def _combine_visibility_score(label: str, window_visibility: float, local_visibility: float) -> float:
        local_weight = {
            "ESP32": 0.28,
            "USB_PORT": 0.72,
            "JST_CONNECTOR": 0.74,
            "RESET_BUTTON": 0.84,
        }.get(label, 0.68)
        window_visibility = max(0.0, float(window_visibility))
        local_visibility = max(0.0, float(local_visibility))
        return float(np.clip(local_weight * local_visibility + (1.0 - local_weight) * window_visibility, 0.0, 1.0))

    @staticmethod
    def _roi_visibility_score(label: str, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = crop if crop.ndim == 2 else np.mean(crop, axis=2).astype(np.uint8)
        if label in {"USB_PORT", "JST_CONNECTOR", "RESET_BUTTON"}:
            clahe = cv.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
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
            shape_score = max(bright_shape, dark_shape, BoardFirstDetector._small_button_shape_score(gray))
            return float(np.clip(0.26 * contrast_score + 0.32 * small_edge_score + 0.42 * shape_score, 0.0, 1.0))
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
    def _small_button_shape_score(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        edges = cv.Canny(blurred, 45, 135)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, gray.shape[0] * gray.shape[1]))
        best = 0.0
        for contour in contours:
            area = float(cv.contourArea(contour))
            area_ratio = area / roi_area
            if area_ratio < 0.004 or area_ratio > 0.22:
                continue
            x, y, w, h = cv.boundingRect(contour)
            if w <= 2 or h <= 2:
                continue
            aspect = w / max(1.0, float(h))
            normalized_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
            if normalized_aspect > 2.25:
                continue
            rectangularity = float(np.clip(area / max(1.0, float(w * h)), 0.0, 1.0))
            extent_score = float(np.exp(-abs(np.log(max(1e-6, area_ratio) / 0.055))))
            aspect_score = float(np.exp(-0.9 * abs(np.log(max(1e-6, normalized_aspect) / 1.20))))
            best = max(best, 0.40 * rectangularity + 0.35 * extent_score + 0.25 * aspect_score)
        return float(np.clip(best, 0.0, 1.0))

    def _update_component_track(
        self,
        spec: ComponentSpec,
        canonical_bbox: BBox,
        score: float,
        visibility_score: float,
        mode: str,
    ) -> None:
        if not self._cfg.enable_tracking:
            return
        previous = self._component_tracks.get(spec.label)
        if previous is not None and mode != "full":
            alpha = float(np.clip(spec.track_smoothing_alpha, 0.05, 1.0))
            canonical_bbox = self._smooth_bbox(previous.canonical_bbox, canonical_bbox, alpha)
            hits = previous.hits + 1
        elif previous is not None:
            hits = previous.hits + 1
        else:
            hits = 1
        if mode == "persistent":
            state = TRACK_LOCAL_SEARCH
        elif hits <= 1 and previous is None:
            state = TRACK_ACQUIRE
        else:
            state = TRACK_LOCKED
        self._component_tracks[spec.label] = _ComponentTrack(
            canonical_bbox=canonical_bbox,
            score=float(score),
            visibility_score=float(visibility_score),
            missing_frames=0,
            hits=hits,
            state=state,
        )
        LOGGER.debug(
            "component track updated: label=%s state=%s mode=%s canonical=%s score=%.3f visibility=%.3f hits=%d",
            spec.label,
            state,
            mode,
            canonical_bbox,
            score,
            visibility_score,
            hits,
        )

    def _mark_component_missing(self, spec: ComponentSpec) -> None:
        if not self._cfg.enable_tracking:
            return
        track = self._component_tracks.get(spec.label)
        if track is None:
            return
        missing = track.missing_frames + 1
        if missing > spec.track_max_missing:
            LOGGER.debug("component track dropped: label=%s state=%s missing=%d", spec.label, TRACK_LOST, missing)
            self._component_tracks.pop(spec.label, None)
            return
        self._component_tracks[spec.label] = _ComponentTrack(
            canonical_bbox=track.canonical_bbox,
            score=track.score * spec.persistence_decay,
            visibility_score=track.visibility_score,
            missing_frames=missing,
            hits=track.hits,
            state=TRACK_LOCAL_SEARCH,
        )
        LOGGER.debug(
            "component track missing: label=%s state=%s missing=%d score=%.3f canonical=%s",
            spec.label,
            TRACK_LOCAL_SEARCH,
            missing,
            track.score * spec.persistence_decay,
            track.canonical_bbox,
        )

    def _locked_search_window(
        self,
        canonical_bbox: BBox,
        roi_bbox: BBox,
        warped_shape: tuple[int, ...],
        spec: ComponentSpec,
    ) -> BBox:
        expanded = self._expand_bbox(canonical_bbox, spec.local_search_expansion, warped_shape)
        return self._intersect_bbox(expanded, roi_bbox) or roi_bbox

    def _track_is_active(self, spec: ComponentSpec, track: _ComponentTrack | None) -> bool:
        return bool(self._cfg.enable_tracking and track is not None and track.missing_frames <= spec.track_max_missing)

    def _keep_score_threshold(self, spec: ComponentSpec) -> float:
        if spec.keep_score_threshold > 0.0:
            return spec.keep_score_threshold
        return max(0.05, spec.score_threshold - 0.08)

    def _keep_min_visibility_score(self, spec: ComponentSpec) -> float:
        if spec.keep_min_visibility_score > 0.0:
            return spec.keep_min_visibility_score
        return max(0.0, spec.min_visibility_score - 0.06)

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

    @staticmethod
    def _smooth_bbox(previous: BBox, current: BBox, alpha: float) -> BBox:
        return BBox(
            int(round(alpha * current.x1 + (1.0 - alpha) * previous.x1)),
            int(round(alpha * current.y1 + (1.0 - alpha) * previous.y1)),
            int(round(alpha * current.x2 + (1.0 - alpha) * previous.x2)),
            int(round(alpha * current.y2 + (1.0 - alpha) * previous.y2)),
        )

    @staticmethod
    def _expand_bbox(bbox: BBox, expansion: float, shape: tuple[int, ...]) -> BBox:
        height, width = shape[:2]
        cx, cy = BoardFirstDetector._bbox_center(bbox)
        half_w = 0.5 * max(1.0, bbox.width()) * (1.0 + 2.0 * max(0.0, expansion))
        half_h = 0.5 * max(1.0, bbox.height()) * (1.0 + 2.0 * max(0.0, expansion))
        return BBox(
            int(max(0, np.floor(cx - half_w))),
            int(max(0, np.floor(cy - half_h))),
            int(min(width, np.ceil(cx + half_w))),
            int(min(height, np.ceil(cy + half_h))),
        )

    @staticmethod
    def _intersect_bbox(a: BBox, b: BBox) -> BBox | None:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox(x1, y1, x2, y2)

    @staticmethod
    def _clip_bbox(bbox: BBox, shape: tuple[int, ...]) -> BBox:
        height, width = shape[:2]
        return BBox(
            int(np.clip(bbox.x1, 0, width)),
            int(np.clip(bbox.y1, 0, height)),
            int(np.clip(bbox.x2, 0, width)),
            int(np.clip(bbox.y2, 0, height)),
        )

    @staticmethod
    def _bbox_center(bbox: BBox) -> tuple[float, float]:
        return 0.5 * (bbox.x1 + bbox.x2), 0.5 * (bbox.y1 + bbox.y2)

    @staticmethod
    def _quad_to_bbox(quad: np.ndarray, frame_shape: tuple[int, ...]) -> BBox:
        height, width = frame_shape[:2]
        return BBox(
            int(max(0, np.floor(np.min(quad[:, 0])))),
            int(max(0, np.floor(np.min(quad[:, 1])))),
            int(min(width, np.ceil(np.max(quad[:, 0])))),
            int(min(height, np.ceil(np.max(quad[:, 1])))),
        )

    @staticmethod
    def _normalized_quad_shift(current: np.ndarray, previous: np.ndarray, previous_bbox: BBox) -> float:
        diag = max(1.0, float(np.hypot(previous_bbox.width(), previous_bbox.height())))
        return float(np.mean(np.linalg.norm(current.astype(np.float32) - previous.astype(np.float32), axis=1)) / diag)

    @staticmethod
    def _quad_area(quad: np.ndarray) -> float:
        return float(abs(cv.contourArea(quad.astype(np.float32))))

    def _pose_rejection_reason(self, current: BoardLocalization, previous: BoardLocalization) -> str | None:
        current_area = max(1.0, self._quad_area(current.quad))
        previous_area = max(1.0, self._quad_area(previous.quad))
        area_growth = current_area / previous_area - 1.0
        quality_drop = previous.warp_quality_score - current.warp_quality_score
        tightness_drop = previous.tightness_score - current.tightness_score
        shift = self._normalized_quad_shift(current.quad, previous.quad, previous.bbox)

        if (
            quality_drop > self._cfg.board_pose_max_quality_drop
            and current.score + self._cfg.board_pose_quality_margin < previous.score
        ):
            return "quality_drop"
        if (
            area_growth > self._cfg.board_pose_max_area_growth
            and tightness_drop > 0.06
        ):
            return "loose_area_growth"
        if (
            tightness_drop > self._cfg.board_pose_max_tightness_drop
            and current.warp_quality_score + self._cfg.board_pose_quality_margin < previous.warp_quality_score
        ):
            return "tightness_drop"
        if (
            shift > 2.0 * self._cfg.board_smoothing_max_shift
            and (area_growth > 0.08 or quality_drop > 0.08 or tightness_drop > 0.08)
            and current.warp_quality_score < previous.warp_quality_score + self._cfg.board_pose_quality_margin
            and current.tightness_score < previous.tightness_score + 0.04
        ):
            return "large_shift_without_quality_gain"
        return None

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

    def _tracked_board_rescue(self, frame: np.ndarray, tracking_hint: BBox | None) -> BoardLocalization | None:
        if not self._cfg.enable_tracking or tracking_hint is None or self._last_localization is None:
            return None
        if not hasattr(self._localizer, "localize_tracked_rescue"):
            return None

        last = self._last_localization
        try:
            rescued = self._localizer.localize_tracked_rescue(
                frame,
                tracking_hint,
                previous_score=last.score,
                previous_warp_quality=last.warp_quality_score,
            )
        except TypeError:
            return None
        if rescued is None:
            return None

        shift = self._normalized_quad_shift(rescued.quad, last.quad, last.bbox)
        if shift > 2.2 * self._cfg.board_smoothing_max_shift:
            LOGGER.debug(
                "board tracked rescue discarded after pose check: shift=%.3f max=%.3f bbox=%s previous=%s",
                shift,
                2.2 * self._cfg.board_smoothing_max_shift,
                rescued.bbox,
                last.bbox,
            )
            return None
        LOGGER.debug(
            "board tracked rescue candidate accepted by detector: shift=%.3f score=%.3f "
            "warp_quality=%.3f identity=%.3f skin=%.3f previous_score=%.3f previous_warp=%.3f",
            shift,
            rescued.score,
            rescued.warp_quality_score,
            rescued.board_identity_score,
            rescued.skin_ratio,
            last.score,
            last.warp_quality_score,
        )
        return rescued

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
        mode: str = "full",
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "component accepted: label=%s mode=%s score=%.3f board_score=%.3f warp_quality=%.3f "
            "match=%.3f second=%.3f margin=%.3f visibility=%.3f visibility_weight=%.3f warp_weight=%.3f",
            spec.label,
            mode,
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
