"""Board localization and perspective normalization.

This module finds candidate PCB quadrilaterals, scores board evidence, warps the
board into canonical coordinates, and normalizes orientation for component ROIs.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- logging: https://docs.python.org/3/library/logging.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.preprocessing.color import normalize_gray, to_gray
from src.preprocessing.filters import clahe_gray, gaussian_blur
from src.utils.types import BBox

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardWarpConfig:
    """Configuration for board localisation and perspective normalization."""
    output_size: tuple[int, int] = (900, 460)
    blur_ksize: int = 5
    canny_t1: int = 45
    canny_t2: int = 135
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.42
    min_rectangularity: float = 0.55
    expected_aspect_ratio: float = 900.0 / 460.0
    min_aspect_ratio: float = 1.35
    max_aspect_ratio: float = 2.65
    border_margin: int = 8
    close_kernel: int = 5
    open_kernel: int = 5
    search_expansion: float = 1.45
    min_score: float = 0.42
    min_tracked_score: float = 0.34
    min_warp_quality_score: float = 0.38
    min_tracked_warp_quality_score: float = 0.32
    min_pcb_structure_score: float = 0.12
    min_canonical_structure_score: float = 0.0
    min_edge_grid_score: float = 0.0
    min_tightness_score: float = 0.35
    refine_pad_x_ratio: float = 0.035
    refine_pad_y_ratio: float = 0.045
    refine_pad_right_ratio: float = 0.045
    refine_pad_right_connector_ratio: float = 0.078
    refine_connector_score_threshold: float = 0.26
    max_skin_ratio: float = 0.18
    verify_gray_weight: float = 0.65
    verify_edge_weight: float = 0.35
    verify_resize_width: int = 300
    min_objectness_score: float = 0.30
    geometry_weight: float = 0.45
    verify_weight: float = 0.25
    objectness_weight: float = 0.30


@dataclass(frozen=True)
class BoardLocalization:
    """Structured result returned by the board localizer."""
    quad: np.ndarray
    bbox: BBox
    homography: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray
    score: float
    warp_quality_score: float = 1.0
    geometry_score: float = 1.0
    verify_score: float = 1.0
    objectness_score: float = 1.0
    pcb_structure_score: float = 1.0
    tightness_score: float = 1.0
    canonical_structure_score: float = 1.0
    edge_grid_score: float = 1.0
    board_identity_score: float = 1.0
    header_score: float = 1.0
    corner_hole_score: float = 1.0
    connector_score: float = 1.0
    skin_ratio: float = 0.0


@dataclass(frozen=True)
class _BoardCandidate:
    quad: np.ndarray
    bbox: BBox
    geometry_score: float
    verify_score: float
    objectness_score: float
    warp_quality_score: float
    pcb_structure_score: float
    canonical_structure_score: float
    edge_grid_score: float
    board_identity_score: float
    header_score: float
    corner_hole_score: float
    connector_score: float
    tightness_score: float
    skin_ratio: float
    score: float
    homography: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray


@dataclass(frozen=True)
class _RejectedBoardCandidate:
    candidate: _BoardCandidate
    reason: str


@dataclass(frozen=True)
class _BoardSearchResult:
    best: _BoardCandidate | None
    candidate_count: int
    best_rejected: _RejectedBoardCandidate | None = None


@dataclass(frozen=True)
class _StructureMetrics:
    score: float
    skin_ratio: float
    canonical_score: float
    header_score: float
    header_hole_score: float
    corner_hole_score: float
    module_score: float
    connector_score: float
    identity_score: float
    edge_density_score: float
    edge_grid_score: float
    color_score: float


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_quad_long_edge_first(pts: np.ndarray) -> np.ndarray:
    """
    Reorder a quadrilateral so that the top and bottom edges correspond to the long side.

    This avoids warping portrait-oriented board observations into a distorted canonical
    view and makes the board orientation ambiguity only a 180-degree problem.
    """
    quad = order_quad_points(pts)
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])

    if 0.5 * (top + bottom) >= 0.5 * (left + right):
        return quad

    return np.array([quad[3], quad[0], quad[1], quad[2]], dtype=np.float32)


def quad_to_bbox(quad: np.ndarray, frame_shape: tuple[int, ...]) -> BBox:
    """Convert a quadrilateral into a clipped axis-aligned bounding box."""
    h, w = frame_shape[:2]
    x1 = int(max(0, np.floor(np.min(quad[:, 0]))))
    y1 = int(max(0, np.floor(np.min(quad[:, 1]))))
    x2 = int(min(w - 1, np.ceil(np.max(quad[:, 0]))))
    y2 = int(min(h - 1, np.ceil(np.max(quad[:, 1]))))
    return BBox(x1, y1, x2, y2)


def expand_bbox(bbox: BBox, frame_shape: tuple[int, ...], scale: float) -> BBox:
    """Expand a box around its center while keeping it inside the frame."""
    h, w = frame_shape[:2]
    cx = 0.5 * (bbox.x1 + bbox.x2)
    cy = 0.5 * (bbox.y1 + bbox.y2)
    half_w = 0.5 * max(1.0, bbox.width()) * scale
    half_h = 0.5 * max(1.0, bbox.height()) * scale
    x1 = int(max(0, np.floor(cx - half_w)))
    y1 = int(max(0, np.floor(cy - half_h)))
    x2 = int(min(w, np.ceil(cx + half_w)))
    y2 = int(min(h, np.ceil(cy + half_h)))
    return BBox(x1, y1, x2, y2)


def _aspect_ratio(quad: np.ndarray) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    width = max(1e-6, 0.5 * (top + bottom))
    height = max(1e-6, 0.5 * (left + right))
    ratio = width / height
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _border_touch_count(bbox: BBox, frame_shape: tuple[int, ...], margin: int) -> int:
    h, w = frame_shape[:2]
    touches = 0
    if bbox.x1 <= margin:
        touches += 1
    if bbox.y1 <= margin:
        touches += 1
    if bbox.x2 >= (w - 1 - margin):
        touches += 1
    if bbox.y2 >= (h - 1 - margin):
        touches += 1
    return touches


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = a.area() + b.area() - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox_center_shift(a: BBox, b: BBox) -> float:
    acx = 0.5 * (a.x1 + a.x2)
    acy = 0.5 * (a.y1 + a.y2)
    bcx = 0.5 * (b.x1 + b.x2)
    bcy = 0.5 * (b.y1 + b.y2)
    diag = max(1.0, float(np.hypot(b.width(), b.height())))
    return float(np.hypot(acx - bcx, acy - bcy) / diag)


def _size_score(area_ratio: float) -> float:
    if area_ratio < 0.01:
        return 0.0
    if area_ratio <= 0.08:
        return area_ratio / 0.08
    if area_ratio <= 0.25:
        return 1.0
    if area_ratio <= 0.42:
        return 1.0 - (area_ratio - 0.25) / 0.17
    return 0.0


def _rotation_180_matrix(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [-1.0, 0.0, width - 1.0],
            [0.0, -1.0, height - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class BoardLocalizer:
    """
    Detect the FireBeetle board in a frame and warp it into a canonical view.

    The algorithm is deliberately classical and lightweight:
    - generate contour candidates from both edges and dark-object masks
    - score them using geometry and optional board-template verification
    - search near the previous board position first to stabilize webcam usage
    """

    def __init__(self, cfg: BoardWarpConfig, reference_boards: list[np.ndarray] | None = None) -> None:
        self._cfg = cfg
        self._references = [self._prepare_reference(image) for image in (reference_boards or [])]

    def localize(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
        *,
        include_full_frame: bool = True,
    ) -> BoardLocalization | None:
        search_boxes: list[BBox] = []
        min_score = self._cfg.min_score
        min_warp_quality = self._cfg.min_warp_quality_score
        if hint_bbox is not None:
            search_boxes.append(expand_bbox(hint_bbox, frame.shape, self._cfg.search_expansion))
            min_score = self._cfg.min_tracked_score
            min_warp_quality = self._cfg.min_tracked_warp_quality_score
        if include_full_frame or not search_boxes:
            search_boxes.append(BBox(0, 0, frame.shape[1], frame.shape[0]))

        best: _BoardCandidate | None = None
        best_rejected: _RejectedBoardCandidate | None = None
        total_candidates = 0
        for search_box in search_boxes:
            result = self._find_best_candidate(frame, search_box)
            total_candidates += result.candidate_count
            if result.best_rejected is not None:
                best_rejected = self._better_rejected_candidate(best_rejected, result.best_rejected)
            candidate = result.best
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
            if candidate.score >= min_score:
                break

        if best is None:
            self._log_board_rejection(
                "all_candidates_rejected" if total_candidates > 0 else "no_candidate",
                best_rejected,
                candidate_found=total_candidates > 0,
                candidate_count=total_candidates,
                hint=hint_bbox is not None,
                include_full_frame=include_full_frame,
                min_score=min_score,
                min_warp_quality=min_warp_quality,
            )
            return None

        if best.score < min_score or best.warp_quality_score < min_warp_quality:
            reasons: list[str] = []
            if best.score < min_score:
                reasons.append("low_score")
            if best.warp_quality_score < min_warp_quality:
                reasons.append("low_warp_quality")
            self._log_board_rejection(
                "+".join(reasons),
                _RejectedBoardCandidate(best, "+".join(reasons)),
                candidate_found=True,
                candidate_count=total_candidates,
                hint=hint_bbox is not None,
                include_full_frame=include_full_frame,
                min_score=min_score,
                min_warp_quality=min_warp_quality,
            )
            return None

        LOGGER.debug(
            "board candidate won: score=%.3f min_score=%.3f warp_quality=%.3f min_warp=%.3f "
            "geometry=%.3f verify=%.3f objectness=%.3f structure=%.3f identity=%.3f "
            "header=%.3f corner=%.3f connector=%.3f edge_grid=%.3f skin=%.3f "
            "bbox=%s candidates=%d hint=%s include_full_frame=%s",
            best.score,
            min_score,
            best.warp_quality_score,
            min_warp_quality,
            best.geometry_score,
            best.verify_score,
            best.objectness_score,
            best.pcb_structure_score,
            best.board_identity_score,
            best.header_score,
            best.corner_hole_score,
            best.connector_score,
            best.edge_grid_score,
            best.skin_ratio,
            best.bbox,
            total_candidates,
            hint_bbox is not None,
            include_full_frame,
        )

        return BoardLocalization(
            quad=best.quad,
            bbox=best.bbox,
            homography=best.homography,
            h_inv=best.h_inv,
            warped=best.warped,
            score=best.score,
            warp_quality_score=best.warp_quality_score,
            geometry_score=best.geometry_score,
            verify_score=best.verify_score,
            objectness_score=best.objectness_score,
            pcb_structure_score=best.pcb_structure_score,
            tightness_score=best.tightness_score,
            canonical_structure_score=best.canonical_structure_score,
            edge_grid_score=best.edge_grid_score,
            board_identity_score=best.board_identity_score,
            header_score=best.header_score,
            corner_hole_score=best.corner_hole_score,
            connector_score=best.connector_score,
            skin_ratio=best.skin_ratio,
        )

    def localize_tracked_rescue(
        self,
        frame: np.ndarray,
        previous_bbox: BBox,
        *,
        previous_score: float,
        previous_warp_quality: float,
    ) -> BoardLocalization | None:
        """
        Accept a near-miss board only when it stays close to a strong previous pose.

        This is intentionally narrower than normal localization: it is for live
        streams where one later validation metric flickers while the board pose is
        still geometrically plausible and close to the last good detection.
        """
        if previous_score < 0.58 or previous_warp_quality < 0.56:
            return None

        search_box = expand_bbox(previous_bbox, frame.shape, self._cfg.search_expansion)
        result = self._find_best_candidate(frame, search_box)
        rejected = result.best_rejected
        if result.best is not None:
            threshold_reason = self._tracked_threshold_rejection_reason(result.best)
            rejected = _RejectedBoardCandidate(result.best, threshold_reason) if threshold_reason is not None else rejected
        if rejected is None:
            return None

        candidate = rejected.candidate
        rescue_reason = self._tracked_rescue_rejection_reason(candidate, rejected.reason, previous_bbox)
        if rescue_reason is not None:
            LOGGER.debug(
                "board tracked rescue rejected: reason=%s original_reason=%s "
                "score=%.3f geometry=%.3f objectness=%.3f structure=%.3f canonical=%.3f "
                "identity=%.3f header=%.3f corner=%.3f connector=%.3f edge_grid=%.3f "
                "tightness=%.3f warp_quality=%.3f skin=%.3f bbox=%s previous_bbox=%s",
                rescue_reason,
                rejected.reason,
                candidate.score,
                candidate.geometry_score,
                candidate.objectness_score,
                candidate.pcb_structure_score,
                candidate.canonical_structure_score,
                candidate.board_identity_score,
                candidate.header_score,
                candidate.corner_hole_score,
                candidate.connector_score,
                candidate.edge_grid_score,
                candidate.tightness_score,
                candidate.warp_quality_score,
                candidate.skin_ratio,
                candidate.bbox,
                previous_bbox,
            )
            return None

        LOGGER.debug(
            "board tracked rescue accepted: original_reason=%s score=%.3f geometry=%.3f "
            "objectness=%.3f structure=%.3f canonical=%.3f identity=%.3f header=%.3f "
            "corner=%.3f connector=%.3f edge_grid=%.3f tightness=%.3f warp_quality=%.3f "
            "skin=%.3f bbox=%s previous_bbox=%s",
            rejected.reason,
            candidate.score,
            candidate.geometry_score,
            candidate.objectness_score,
            candidate.pcb_structure_score,
            candidate.canonical_structure_score,
            candidate.board_identity_score,
            candidate.header_score,
            candidate.corner_hole_score,
            candidate.connector_score,
            candidate.edge_grid_score,
            candidate.tightness_score,
            candidate.warp_quality_score,
            candidate.skin_ratio,
            candidate.bbox,
            previous_bbox,
        )
        return BoardLocalization(
            quad=candidate.quad,
            bbox=candidate.bbox,
            homography=candidate.homography,
            h_inv=candidate.h_inv,
            warped=candidate.warped,
            score=max(candidate.score, self._cfg.min_tracked_score),
            warp_quality_score=max(candidate.warp_quality_score, self._cfg.min_tracked_warp_quality_score),
            geometry_score=candidate.geometry_score,
            verify_score=candidate.verify_score,
            objectness_score=candidate.objectness_score,
            pcb_structure_score=candidate.pcb_structure_score,
            tightness_score=candidate.tightness_score,
            canonical_structure_score=candidate.canonical_structure_score,
            edge_grid_score=candidate.edge_grid_score,
            board_identity_score=candidate.board_identity_score,
            header_score=candidate.header_score,
            corner_hole_score=candidate.corner_hole_score,
            connector_score=candidate.connector_score,
            skin_ratio=candidate.skin_ratio,
        )

    def _find_best_candidate(self, frame: np.ndarray, search_box: BBox) -> _BoardSearchResult:
        crop = frame[search_box.y1:search_box.y2, search_box.x1:search_box.x2]
        if crop.size == 0:
            return _BoardSearchResult(None, 0)

        gray = normalize_gray(to_gray(crop))
        gray = clahe_gray(gray)
        gray = gaussian_blur(gray, self._cfg.blur_ksize)

        candidates = self._candidate_quads(gray, search_box, frame.shape)
        if not candidates:
            return _BoardSearchResult(None, 0)

        out_w, out_h = self._cfg.output_size
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        best: _BoardCandidate | None = None
        best_rejected: _RejectedBoardCandidate | None = None
        for quad, geometry_score in candidates:
            homography = cv.getPerspectiveTransform(quad, dst)
            warped = cv.warpPerspective(frame, homography, (out_w, out_h))
            warped, homography = self._normalize_orientation(warped, homography)
            warped, homography, quad, tightness_score = self._refine_warp_tightness(frame, warped, homography, quad)
            verify_score = self._verify_board(warped)
            objectness_score = self._board_objectness_score(warped)
            structure = self._pcb_structure_metrics(warped)
            warp_quality_score = self._warp_quality_score(
                quad,
                warped,
                geometry_score,
                verify_score,
                objectness_score,
                structure.score,
                tightness_score,
            )

            if self._references:
                base_score = (
                    self._cfg.geometry_weight * geometry_score
                    + self._cfg.verify_weight * max(0.0, verify_score)
                    + self._cfg.objectness_weight * objectness_score
                )
            else:
                base_score = 0.65 * geometry_score + 0.35 * objectness_score
            bbox = quad_to_bbox(quad, frame.shape)
            score = self._candidate_confidence_score(
                base_score,
                warp_quality_score,
                verify_score,
                structure,
                bbox,
                frame.shape,
            )
            candidate = _BoardCandidate(
                quad=quad,
                bbox=bbox,
                geometry_score=geometry_score,
                verify_score=verify_score,
                objectness_score=objectness_score,
                warp_quality_score=warp_quality_score,
                pcb_structure_score=structure.score,
                canonical_structure_score=structure.canonical_score,
                edge_grid_score=structure.edge_grid_score,
                board_identity_score=structure.identity_score,
                header_score=structure.header_score,
                corner_hole_score=structure.corner_hole_score,
                connector_score=structure.connector_score,
                tightness_score=tightness_score,
                skin_ratio=structure.skin_ratio,
                score=score,
                homography=homography,
                h_inv=np.linalg.inv(homography),
                warped=warped,
            )

            rejection_reason = self._candidate_rejection_reason(candidate, structure, frame.shape)
            if rejection_reason is not None:
                rejected = _RejectedBoardCandidate(candidate, rejection_reason)
                best_rejected = self._better_rejected_candidate(best_rejected, rejected)
                self._log_board_candidate_rejection(rejected, structure)
                continue

            if best is None or candidate.score > best.score:
                best = candidate

            self._log_board_candidate_acceptance(candidate, structure, frame.shape)

        return _BoardSearchResult(best, len(candidates), best_rejected)

    def _candidate_quads(
        self,
        gray: np.ndarray,
        search_box: BBox,
        frame_shape: tuple[int, ...],
    ) -> list[tuple[np.ndarray, float]]:
        h_roi, w_roi = gray.shape[:2]
        roi_area = float(max(1, h_roi * w_roi))

        edges = cv.Canny(gray, self._cfg.canny_t1, self._cfg.canny_t2)
        close_kernel = np.ones((max(3, self._cfg.close_kernel), max(3, self._cfg.close_kernel)), np.uint8)
        open_kernel = np.ones((max(3, self._cfg.open_kernel), max(3, self._cfg.open_kernel)), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, close_kernel, iterations=1)

        _, dark_mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        dark_mask = cv.morphologyEx(dark_mask, cv.MORPH_CLOSE, close_kernel, iterations=1)
        dark_mask = cv.morphologyEx(dark_mask, cv.MORPH_OPEN, open_kernel, iterations=1)

        contours: list[np.ndarray] = []
        contours_edges, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_mask, _ = cv.findContours(dark_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours.extend(contours_edges)
        contours.extend(contours_mask)

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)
        results: list[tuple[np.ndarray, float]] = []
        seen: set[tuple[int, int, int, int]] = set()

        for contour in contours:
            contour_area = float(cv.contourArea(contour))
            if contour_area < 0.01 * roi_area:
                continue

            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                quad = order_quad_long_edge_first(approx)
            else:
                rect = cv.minAreaRect(contour)
                quad = order_quad_long_edge_first(cv.boxPoints(rect))

            quad[:, 0] += search_box.x1
            quad[:, 1] += search_box.y1
            bbox = quad_to_bbox(quad, frame_shape)
            if bbox.area() <= 0:
                continue

            area_ratio = contour_area / frame_area
            if area_ratio < self._cfg.min_area_ratio or area_ratio > self._cfg.max_area_ratio:
                continue

            if _border_touch_count(bbox, frame_shape, self._cfg.border_margin) >= 2:
                continue

            xs = quad[:, 0]
            ys = quad[:, 1]
            box_area = max(1.0, float((xs.max() - xs.min()) * (ys.max() - ys.min())))
            rectangularity = float(contour_area) / box_area
            if rectangularity < self._cfg.min_rectangularity:
                continue

            aspect = _aspect_ratio(quad)
            if aspect < self._cfg.min_aspect_ratio or aspect > self._cfg.max_aspect_ratio:
                continue

            aspect_penalty = abs(np.log(aspect / self._cfg.expected_aspect_ratio))
            aspect_score = float(np.exp(-2.6 * aspect_penalty))
            geometry_score = 0.50 * aspect_score + 0.30 * rectangularity + 0.20 * _size_score(area_ratio)

            key = (
                int(round(bbox.x1 / 8)),
                int(round(bbox.y1 / 8)),
                int(round(bbox.x2 / 8)),
                int(round(bbox.y2 / 8)),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append((quad.astype(np.float32), float(geometry_score)))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:8]

    def _candidate_confidence_score(
        self,
        base_score: float,
        warp_quality_score: float,
        verify_score: float,
        structure: _StructureMetrics,
        bbox: BBox,
        frame_shape: tuple[int, ...],
    ) -> float:
        identity = float(np.clip(structure.identity_score, 0.0, 1.0))
        score = 0.66 * float(np.clip(base_score, 0.0, 1.0)) + 0.17 * warp_quality_score + 0.17 * identity
        score -= self._skin_score_penalty(structure.skin_ratio, identity, verify_score, warp_quality_score)
        score -= self._border_clutter_penalty(bbox, frame_shape, structure, verify_score)
        return float(np.clip(score, 0.0, 1.0))

    def _skin_score_penalty(
        self,
        skin_ratio: float,
        identity_score: float,
        verify_score: float,
        warp_quality_score: float,
    ) -> float:
        overage = max(0.0, skin_ratio - self._cfg.max_skin_ratio)
        if overage <= 0.0:
            return 0.0
        reference_support = float(np.clip((verify_score - 0.05) / 0.30, 0.0, 1.0)) if self._references else 0.0
        evidence = float(np.clip(0.58 * identity_score + 0.24 * warp_quality_score + 0.18 * reference_support, 0.0, 1.0))
        return float(np.clip(overage * (0.24 - 0.18 * evidence), 0.0, 0.18))

    def _border_clutter_penalty(
        self,
        bbox: BBox,
        frame_shape: tuple[int, ...],
        structure: _StructureMetrics,
        verify_score: float,
    ) -> float:
        touches = _border_touch_count(bbox, frame_shape, self._cfg.border_margin)
        if touches <= 0:
            return 0.0
        identity = structure.identity_score
        if identity >= 0.50 or verify_score >= 0.24 or structure.header_score >= 0.50:
            return 0.0
        weak_layout = max(0.0, 0.50 - identity) + max(0.0, 0.35 - structure.edge_grid_score)
        return float(np.clip(0.035 * touches + 0.045 * weak_layout, 0.0, 0.14))

    def _candidate_rejection_reason(
        self,
        candidate: _BoardCandidate,
        structure: _StructureMetrics,
        frame_shape: tuple[int, ...],
    ) -> str | None:
        if candidate.objectness_score < self._cfg.min_objectness_score:
            return "low_objectness"
        skin_reason = self._skin_hard_rejection_reason(candidate, structure)
        if skin_reason is not None:
            return skin_reason
        if structure.canonical_score < self._cfg.min_canonical_structure_score:
            return "low_canonical_structure"
        if structure.edge_grid_score < self._cfg.min_edge_grid_score:
            return "low_edge_distribution"
        identity_reason = self._weak_identity_rejection_reason(candidate, structure, frame_shape)
        if identity_reason is not None:
            return identity_reason
        if structure.score < self._cfg.min_pcb_structure_score:
            return "low_pcb_structure"
        if candidate.tightness_score < self._cfg.min_tightness_score:
            return "loose_warp"
        return None

    def _skin_hard_rejection_reason(
        self,
        candidate: _BoardCandidate,
        structure: _StructureMetrics,
    ) -> str | None:
        if candidate.skin_ratio <= self._cfg.max_skin_ratio:
            return None

        evidence = self._board_evidence_score(candidate, structure)
        overage = candidate.skin_ratio - self._cfg.max_skin_ratio
        if (
            candidate.skin_ratio >= 0.72
            and evidence < 0.78
        ):
            return "skin_dominant_weak_pcb"
        if (
            overage > 0.22
            and evidence < 0.66
        ):
            return "skin_like_region"
        if (
            overage > 0.08
            and evidence < 0.50
        ):
            return "skin_like_region"
        if (
            overage > 0.025
            and evidence < 0.38
        ):
            return "skin_like_region"

        LOGGER.debug(
            "board skin gate converted to penalty: skin=%.3f max=%.3f evidence=%.3f "
            "identity=%.3f header=%.3f corner=%.3f connector=%.3f geometry=%.3f "
            "verify=%.3f objectness=%.3f warp_quality=%.3f",
            candidate.skin_ratio,
            self._cfg.max_skin_ratio,
            evidence,
            structure.identity_score,
            structure.header_score,
            structure.corner_hole_score,
            structure.connector_score,
            candidate.geometry_score,
            candidate.verify_score,
            candidate.objectness_score,
            candidate.warp_quality_score,
        )
        return None

    def _weak_identity_rejection_reason(
        self,
        candidate: _BoardCandidate,
        structure: _StructureMetrics,
        frame_shape: tuple[int, ...],
    ) -> str | None:
        if self._cfg.min_canonical_structure_score <= 0.0 and self._cfg.min_edge_grid_score <= 0.0 and not self._references:
            return None

        reference_support = candidate.verify_score >= 0.24
        strong_cues = sum(
            [
                structure.header_score >= 0.46,
                structure.corner_hole_score >= 0.34,
                structure.connector_score >= 0.46,
                structure.module_score >= 0.48,
                structure.edge_grid_score >= 0.48,
                reference_support,
            ]
        )
        border_touches = _border_touch_count(candidate.bbox, frame_shape, self._cfg.border_margin)

        if (
            border_touches > 0
            and structure.identity_score < 0.46
            and structure.edge_grid_score < 0.36
            and not reference_support
        ):
            return "background_border_clutter"
        if (
            structure.identity_score < 0.32
            and strong_cues < 2
            and candidate.verify_score < 0.18
        ):
            return "weak_pcb_identity"
        if (
            structure.score < max(self._cfg.min_pcb_structure_score + 0.12, 0.30)
            and strong_cues < 2
            and candidate.score < 0.60
        ):
            return "weak_pcb_identity"
        return None

    @staticmethod
    def _board_evidence_score(candidate: _BoardCandidate, structure: _StructureMetrics) -> float:
        verify_support = float(np.clip((candidate.verify_score - 0.02) / 0.26, 0.0, 1.0))
        layout_support = max(structure.header_score, structure.corner_hole_score, structure.connector_score)
        return float(
            np.clip(
                0.30 * structure.identity_score
                + 0.18 * candidate.objectness_score
                + 0.16 * candidate.warp_quality_score
                + 0.14 * layout_support
                + 0.12 * candidate.tightness_score
                + 0.10 * verify_support,
                0.0,
                1.0,
            )
        )

    def _tracked_rescue_rejection_reason(
        self,
        candidate: _BoardCandidate,
        original_reason: str,
        previous_bbox: BBox,
    ) -> str | None:
        if original_reason not in {
            "low_objectness",
            "skin_like_region",
            "skin_dominant_weak_pcb",
            "low_canonical_structure",
            "low_edge_distribution",
            "low_pcb_structure",
            "weak_pcb_identity",
            "background_border_clutter",
            "loose_warp",
            "low_score",
            "low_warp_quality",
        }:
            return "unsupported_reason"

        center_shift = _bbox_center_shift(candidate.bbox, previous_bbox)
        overlap = _bbox_iou(candidate.bbox, previous_bbox)
        area_ratio = candidate.bbox.area() / max(1.0, float(previous_bbox.area()))
        if center_shift > 0.12 and overlap < 0.42:
            return "not_near_previous_pose"
        if area_ratio < 0.62 or area_ratio > 1.48:
            return "area_change_too_large"
        if candidate.geometry_score < 0.52:
            return "weak_geometry"

        objectness_floor = self._cfg.min_objectness_score - 0.08
        if candidate.objectness_score < objectness_floor:
            return "objectness_too_low"
        if candidate.pcb_structure_score < max(0.08, self._cfg.min_pcb_structure_score - 0.06):
            return "structure_too_low"
        if candidate.tightness_score < max(0.26, self._cfg.min_tightness_score - 0.12):
            return "tightness_too_low"

        if original_reason == "skin_like_region":
            if not (
                candidate.skin_ratio <= max(self._cfg.max_skin_ratio + 0.30, 0.42)
                and candidate.board_identity_score >= 0.54
                and candidate.objectness_score >= 0.60
                and candidate.warp_quality_score >= self._cfg.min_tracked_warp_quality_score
            ):
                return "skin_evidence_too_weak"
        elif original_reason == "skin_dominant_weak_pcb":
            return "skin_dominant"
        elif original_reason == "low_objectness":
            if not (
                candidate.objectness_score >= objectness_floor
                and (candidate.verify_score >= 0.22 or candidate.board_identity_score >= 0.42)
            ):
                return "objectness_rescue_unsupported"
        elif original_reason in {"weak_pcb_identity", "background_border_clutter"}:
            if not (
                candidate.board_identity_score >= 0.50
                and candidate.header_score >= 0.34
                and candidate.edge_grid_score >= 0.34
            ):
                return "identity_rescue_unsupported"
        elif original_reason == "low_canonical_structure":
            if candidate.canonical_structure_score < self._cfg.min_canonical_structure_score - 0.04:
                return "canonical_too_low"
        elif original_reason == "low_edge_distribution":
            if candidate.edge_grid_score < self._cfg.min_edge_grid_score - 0.04:
                return "edge_grid_too_low"
        elif original_reason == "low_pcb_structure":
            if candidate.pcb_structure_score < self._cfg.min_pcb_structure_score - 0.05:
                return "pcb_structure_too_low"
        elif original_reason == "loose_warp":
            if candidate.warp_quality_score < self._cfg.min_tracked_warp_quality_score:
                return "warp_quality_too_low"

        if candidate.score < self._cfg.min_tracked_score - 0.05:
            return "score_too_low"
        if candidate.warp_quality_score < self._cfg.min_tracked_warp_quality_score - 0.04:
            return "warp_quality_too_low"
        return None

    def _tracked_threshold_rejection_reason(self, candidate: _BoardCandidate) -> str | None:
        if candidate.score < self._cfg.min_tracked_score:
            return "low_score"
        if candidate.warp_quality_score < self._cfg.min_tracked_warp_quality_score:
            return "low_warp_quality"
        return None

    @staticmethod
    def _better_rejected_candidate(
        current: _RejectedBoardCandidate | None,
        incoming: _RejectedBoardCandidate | None,
    ) -> _RejectedBoardCandidate | None:
        if incoming is None:
            return current
        if current is None:
            return incoming

        def rank(rejected: _RejectedBoardCandidate) -> float:
            candidate = rejected.candidate
            return (
                0.34 * candidate.score
                + 0.22 * candidate.geometry_score
                + 0.18 * candidate.objectness_score
                + 0.14 * candidate.board_identity_score
                + 0.07 * candidate.pcb_structure_score
                + 0.05 * candidate.warp_quality_score
            )

        return incoming if rank(incoming) > rank(current) else current

    def _log_board_rejection(
        self,
        reason: str,
        rejected: _RejectedBoardCandidate | None,
        *,
        candidate_found: bool,
        candidate_count: int,
        hint: bool,
        include_full_frame: bool,
        min_score: float,
        min_warp_quality: float,
    ) -> None:
        if rejected is None:
            LOGGER.debug(
                "board rejected: reason=%s candidate_found=%s candidate_count=%d hint=%s "
                "include_full_frame=%s min_score=%.3f min_warp_quality=%.3f",
                reason,
                candidate_found,
                candidate_count,
                hint,
                include_full_frame,
                min_score,
                min_warp_quality,
            )
            return

        candidate = rejected.candidate
        LOGGER.debug(
            "board rejected: reason=%s final_candidate_reason=%s candidate_found=%s candidate_count=%d "
            "best_score=%.3f min_score=%.3f geometry=%.3f verify=%.3f objectness=%.3f "
            "structure=%.3f canonical=%.3f identity=%.3f header=%.3f corner=%.3f "
            "connector=%.3f edge_grid=%.3f tightness=%.3f warp_quality=%.3f "
            "min_warp_quality=%.3f skin=%.3f max_skin=%.3f bbox=%s hint=%s include_full_frame=%s",
            reason,
            rejected.reason,
            candidate_found,
            candidate_count,
            candidate.score,
            min_score,
            candidate.geometry_score,
            candidate.verify_score,
            candidate.objectness_score,
            candidate.pcb_structure_score,
            candidate.canonical_structure_score,
            candidate.board_identity_score,
            candidate.header_score,
            candidate.corner_hole_score,
            candidate.connector_score,
            candidate.edge_grid_score,
            candidate.tightness_score,
            candidate.warp_quality_score,
            min_warp_quality,
            candidate.skin_ratio,
            self._cfg.max_skin_ratio,
            candidate.bbox,
            hint,
            include_full_frame,
        )

    def _log_board_candidate_rejection(
        self,
        rejected: _RejectedBoardCandidate,
        structure: _StructureMetrics,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        candidate = rejected.candidate
        LOGGER.debug(
            "board candidate rejected: reason=%s score=%.3f geometry=%.3f verify=%.3f "
            "objectness=%.3f structure=%.3f canonical=%.3f identity=%.3f header=%.3f "
            "header_holes=%.3f corner=%.3f module=%.3f connector=%.3f edge_density=%.3f "
            "edge_grid=%.3f color=%.3f tightness=%.3f warp_quality=%.3f skin=%.3f "
            "max_skin=%.3f bbox=%s",
            rejected.reason,
            candidate.score,
            candidate.geometry_score,
            candidate.verify_score,
            candidate.objectness_score,
            candidate.pcb_structure_score,
            candidate.canonical_structure_score,
            candidate.board_identity_score,
            structure.header_score,
            structure.header_hole_score,
            structure.corner_hole_score,
            structure.module_score,
            structure.connector_score,
            structure.edge_density_score,
            candidate.edge_grid_score,
            structure.color_score,
            candidate.tightness_score,
            candidate.warp_quality_score,
            candidate.skin_ratio,
            self._cfg.max_skin_ratio,
            candidate.bbox,
        )

    def _log_board_candidate_acceptance(
        self,
        candidate: _BoardCandidate,
        structure: _StructureMetrics,
        frame_shape: tuple[int, ...],
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "board candidate accepted: score=%.3f geometry=%.3f verify=%.3f objectness=%.3f "
            "structure=%.3f canonical=%.3f identity=%.3f header=%.3f header_holes=%.3f "
            "corner=%.3f module=%.3f connector=%.3f edge_grid=%.3f tightness=%.3f "
            "warp_quality=%.3f skin=%.3f border_touches=%d bbox=%s",
            candidate.score,
            candidate.geometry_score,
            candidate.verify_score,
            candidate.objectness_score,
            candidate.pcb_structure_score,
            candidate.canonical_structure_score,
            candidate.board_identity_score,
            structure.header_score,
            structure.header_hole_score,
            structure.corner_hole_score,
            structure.module_score,
            structure.connector_score,
            candidate.edge_grid_score,
            candidate.tightness_score,
            candidate.warp_quality_score,
            candidate.skin_ratio,
            _border_touch_count(candidate.bbox, frame_shape, self._cfg.border_margin),
            candidate.bbox,
        )

    def _prepare_reference(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_w, out_h = self._cfg.output_size
        resized = cv.resize(image, (out_w, out_h), interpolation=cv.INTER_AREA)
        resized = self._resize_for_verify(resized)
        gray = clahe_gray(normalize_gray(to_gray(resized)))
        gray = gaussian_blur(gray, 3)
        edges = cv.Canny(gray, self._cfg.canny_t1, self._cfg.canny_t2)
        return gray, edges

    def _verify_board(self, warped: np.ndarray) -> float:
        if not self._references:
            return 1.0

        warped = self._resize_for_verify(warped)
        gray = clahe_gray(normalize_gray(to_gray(warped)))
        gray = gaussian_blur(gray, 3)
        edges = cv.Canny(gray, self._cfg.canny_t1, self._cfg.canny_t2)

        best = -1.0
        for ref_gray, ref_edges in self._references:
            gray_score = float(cv.matchTemplate(gray, ref_gray, cv.TM_CCOEFF_NORMED)[0, 0])
            edge_score = float(cv.matchTemplate(edges, ref_edges, cv.TM_CCOEFF_NORMED)[0, 0])
            combined = self._cfg.verify_gray_weight * gray_score + self._cfg.verify_edge_weight * edge_score
            best = max(best, combined)
        return float(best)

    def _resize_for_verify(self, image: np.ndarray) -> np.ndarray:
        target_width = int(self._cfg.verify_resize_width)
        if target_width <= 0 or image.shape[1] <= target_width:
            return image
        scale = target_width / float(image.shape[1])
        target_height = max(1, int(round(image.shape[0] * scale)))
        return cv.resize(image, (target_width, target_height), interpolation=cv.INTER_AREA)

    def _board_objectness_score(self, warped: np.ndarray) -> float:
        """
        Estimate whether a warped candidate looks like the target PCB.

        This deliberately avoids template identity. A true FireBeetle-style board
        is a dark, edge-rich elongated object; false webcam candidates such as a
        face, wall, shirt logo, or TV edge usually fail at least one of those
        checks even when their geometry looks rectangular.
        """
        gray = normalize_gray(to_gray(warped))
        h, w = gray.shape[:2]
        inner = gray[int(0.06 * h) : int(0.94 * h), int(0.04 * w) : int(0.96 * w)]
        if inner.size == 0:
            return 0.0

        dark_ratio = float(np.mean(inner < 120))
        dark_score = float(np.clip((dark_ratio - 0.18) / 0.42, 0.0, 1.0))

        edge_gray = clahe_gray(gray)
        edges = cv.Canny(edge_gray, self._cfg.canny_t1, self._cfg.canny_t2)
        inner_edges = edges[int(0.06 * h) : int(0.94 * h), int(0.04 * w) : int(0.96 * w)]
        edge_density = float(np.mean(inner_edges > 0)) if inner_edges.size else 0.0
        edge_score = float(np.clip((edge_density - 0.035) / 0.13, 0.0, 1.0))

        return 0.60 * dark_score + 0.40 * edge_score

    def _refine_warp_tightness(
        self,
        frame: np.ndarray,
        warped: np.ndarray,
        homography: np.ndarray,
        quad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Tighten a plausible but loose board warp using the foreground PCB mask.

        The refinement is intentionally conservative: it only crops inward when a
        coherent dark/saturated PCB-like foreground fills enough of the canonical
        view and keeps the expected board aspect. This helps reject/repair boxes
        that include hand, face, wall, or monitor border around the actual PCB.
        """
        foreground = self._foreground_bbox_in_warp(warped)
        if foreground is None:
            return warped, homography, quad, 0.0

        x, y, bw, bh, _fill_ratio = foreground
        h, w = warped.shape[:2]
        tightness = self._tightness_score_from_bbox(x, y, bw, bh, w, h)

        coverage_x = bw / max(1.0, float(w))
        coverage_y = bh / max(1.0, float(h))
        aspect = bw / max(1.0, float(bh))
        aspect_score = float(np.exp(-2.0 * abs(np.log(max(1e-6, aspect) / self._cfg.expected_aspect_ratio))))
        should_refine = (
            aspect_score >= 0.45
            and coverage_x >= 0.50
            and coverage_y >= 0.50
            and (coverage_x < 0.94 or coverage_y < 0.94)
        )
        if not should_refine:
            return warped, homography, quad, tightness

        connector_score = self._right_connector_preservation_score(warped)
        right_pad_ratio = (
            self._cfg.refine_pad_right_connector_ratio
            if connector_score >= self._cfg.refine_connector_score_threshold
            else self._cfg.refine_pad_right_ratio
        )
        pad_x = int(round(max(0.0, self._cfg.refine_pad_x_ratio) * bw))
        pad_right = int(round(max(0.0, right_pad_ratio) * bw))
        pad_y = int(round(max(0.0, self._cfg.refine_pad_y_ratio) * bh))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w - 1, x + bw + pad_right)
        y2 = min(h - 1, y + bh + pad_y)
        if x2 <= x1 or y2 <= y1:
            return warped, homography, quad, tightness

        canonical_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            source_pts = cv.perspectiveTransform(canonical_pts, np.linalg.inv(homography)).reshape(4, 2)
        except np.linalg.LinAlgError:
            return warped, homography, quad, tightness

        out_w, out_h = self._cfg.output_size
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        refined_h = cv.getPerspectiveTransform(source_pts.astype(np.float32), dst)
        refined_warped = cv.warpPerspective(frame, refined_h, (out_w, out_h))
        refined_foreground = self._foreground_bbox_in_warp(refined_warped)
        if refined_foreground is None:
            return warped, homography, quad, tightness

        rx, ry, rbw, rbh, _ = refined_foreground
        refined_tightness = self._tightness_score_from_bbox(rx, ry, rbw, rbh, out_w, out_h)
        if refined_tightness + 0.03 < tightness:
            return warped, homography, quad, tightness

        LOGGER.debug(
            "board warp refined: tightness %.3f -> %.3f coverage=(%.2f, %.2f) "
            "aspect_score=%.3f connector_preserve=%.3f right_pad=%.3f",
            tightness,
            refined_tightness,
            coverage_x,
            coverage_y,
            aspect_score,
            connector_score,
            pad_right / max(1.0, float(bw)),
        )
        return refined_warped, refined_h, source_pts.astype(np.float32), max(tightness, refined_tightness)

    def _foreground_bbox_in_warp(self, warped: np.ndarray) -> tuple[int, int, int, int, float] | None:
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        edges = cv.Canny(clahe_gray(gray), self._cfg.canny_t1, self._cfg.canny_t2)
        edge_support = cv.dilate(edges, np.ones((5, 5), np.uint8), iterations=1) > 0
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        skin = self._skin_mask(hue, sat, val)

        dark_board = gray < 155
        saturated_pcb = (sat > 45) & (val < 220)
        h, w = gray.shape[:2]
        right_zone = np.zeros_like(dark_board, dtype=bool)
        right_zone[:, int(0.62 * w) :] = True
        connector_like = right_zone & edge_support & (val > 132) & (sat < 155)
        mask = (dark_board | saturated_pcb | connector_like) & ~skin
        mask_u8 = (mask.astype(np.uint8)) * 255
        kernel = np.ones((7, 7), np.uint8)
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, kernel, iterations=2)
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        min_area = 0.025 * h * w
        kept = [contour for contour in contours if cv.contourArea(contour) >= min_area]
        if not kept:
            return None

        points = np.vstack(kept)
        x, y, bw, bh = cv.boundingRect(points)
        fill_ratio = float(np.mean(mask_u8[y : y + bh, x : x + bw] > 0)) if bw > 0 and bh > 0 else 0.0
        return int(x), int(y), int(bw), int(bh), fill_ratio

    def _right_connector_preservation_score(self, warped: np.ndarray) -> float:
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        edges = cv.Canny(clahe_gray(gray), self._cfg.canny_t1, self._cfg.canny_t2)
        h, w = gray.shape[:2]
        region_hsv = hsv[int(0.10 * h) : int(0.88 * h), int(0.64 * w) : int(0.99 * w)]
        region_edges = edges[int(0.10 * h) : int(0.88 * h), int(0.64 * w) : int(0.99 * w)]
        if region_hsv.size == 0 or region_edges.size == 0:
            return 0.0
        bright_low_sat = float(np.mean((region_hsv[:, :, 2] > 140) & (region_hsv[:, :, 1] < 135)))
        edge_density = float(np.mean(region_edges > 0))
        bright_score = float(np.clip((bright_low_sat - 0.025) / 0.20, 0.0, 1.0))
        edge_score = float(np.clip((edge_density - 0.020) / 0.12, 0.0, 1.0))
        return float(np.clip(0.58 * bright_score + 0.42 * edge_score, 0.0, 1.0))

    def _tightness_score_from_bbox(self, x: int, y: int, bw: int, bh: int, width: int, height: int) -> float:
        coverage_x = bw / max(1.0, float(width))
        coverage_y = bh / max(1.0, float(height))
        coverage = float(np.sqrt(max(0.0, coverage_x * coverage_y)))
        coverage_score = float(np.clip((coverage - 0.48) / 0.42, 0.0, 1.0))

        margins = np.array(
            [
                x / max(1.0, float(width)),
                y / max(1.0, float(height)),
                (width - (x + bw)) / max(1.0, float(width)),
                (height - (y + bh)) / max(1.0, float(height)),
            ],
            dtype=np.float32,
        )
        margin_score = float(np.clip(1.0 - np.max(margins) / 0.22, 0.0, 1.0))
        aspect = bw / max(1.0, float(bh))
        aspect_score = float(np.exp(-2.0 * abs(np.log(max(1e-6, aspect) / self._cfg.expected_aspect_ratio))))
        return float(np.clip(0.46 * coverage_score + 0.34 * margin_score + 0.20 * aspect_score, 0.0, 1.0))

    def _skin_mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
        red_or_orange = (hue < 24) | (hue > 168)
        return red_or_orange & (sat > 32) & (sat < 185) & (val > 55) & (val < 245)

    def _pcb_structure_metrics(self, warped: np.ndarray) -> _StructureMetrics:
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        h, w = gray.shape[:2]
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        inner_slice = np.s_[int(0.06 * h) : int(0.94 * h), int(0.04 * w) : int(0.96 * w)]
        inner_hue = hue[inner_slice]
        inner_sat = sat[inner_slice]
        inner_val = val[inner_slice]
        skin_ratio = float(np.mean(self._skin_mask(inner_hue, inner_sat, inner_val))) if inner_hue.size else 0.0

        edge_gray = clahe_gray(gray)
        edges = cv.Canny(edge_gray, self._cfg.canny_t1, self._cfg.canny_t2)
        inner_edges = edges[inner_slice]
        edge_density = float(np.mean(inner_edges > 0)) if inner_edges.size else 0.0
        edge_density_score = float(np.clip((edge_density - 0.025) / 0.12, 0.0, 1.0))
        edge_grid_score = self._edge_grid_score(inner_edges)

        inner_sat = sat[inner_slice]
        inner_val = val[inner_slice]
        dark_ratio = float(np.mean(inner_val < 135)) if inner_val.size else 0.0
        saturated_ratio = float(np.mean((inner_sat > 42) & (inner_val < 220))) if inner_val.size else 0.0
        color_score = float(np.clip((max(dark_ratio, saturated_ratio) - 0.16) / 0.42, 0.0, 1.0))

        top_band = edges[int(0.05 * h) : int(0.24 * h), int(0.04 * w) : int(0.96 * w)]
        bottom_band = edges[int(0.76 * h) : int(0.95 * h), int(0.04 * w) : int(0.96 * w)]

        def header_band_score(band: np.ndarray) -> float:
            if band.size == 0:
                return 0.0
            density = float(np.mean(band > 0))
            column_mass = np.mean(band > 0, axis=0)
            threshold = float(np.mean(column_mass) + 0.60 * np.std(column_mass))
            peak_mask = column_mass > threshold
            peaks = int(np.count_nonzero(peak_mask))
            peak_score = float(np.clip((peaks - 8) / 34.0, 0.0, 1.0))
            centers = np.flatnonzero(peak_mask)
            if centers.size >= 8:
                spacing = np.diff(centers.astype(np.float32))
                spacing = spacing[spacing > 1.0]
                regularity = 0.0
                if spacing.size >= 4:
                    regularity = float(np.clip(1.0 - (np.std(spacing) / max(1.0, np.mean(spacing))) / 1.8, 0.0, 1.0))
                peak_score = 0.75 * peak_score + 0.25 * regularity
            density_score = float(np.clip((density - 0.020) / 0.11, 0.0, 1.0))
            return 0.55 * density_score + 0.45 * peak_score

        top_edge_score = header_band_score(top_band)
        bottom_edge_score = header_band_score(bottom_band)
        header_edge_score = 0.55 * max(top_edge_score, bottom_edge_score) + 0.45 * min(top_edge_score, bottom_edge_score)
        header_hole_score = self._header_hole_band_score(hsv, edges)
        header_score = 0.54 * header_edge_score + 0.46 * header_hole_score

        corner_hole_score = self._corner_hole_pattern_score(hsv, edges)
        canonical_score, module_score, connector_score = self._canonical_region_scores(warped)
        identity_score = float(
            np.clip(
                0.24 * canonical_score
                + 0.18 * module_score
                + 0.18 * connector_score
                + 0.18 * header_score
                + 0.12 * corner_hole_score
                + 0.10 * edge_grid_score,
                0.0,
                1.0,
            )
        )
        skin_penalty = float(np.clip(1.0 - skin_ratio / max(1e-6, self._cfg.max_skin_ratio), 0.0, 1.0))
        score = (
            0.26 * identity_score
            + 0.20 * header_score
            + 0.16 * edge_density_score
            + 0.13 * edge_grid_score
            + 0.10 * canonical_score
            + 0.07 * color_score
            + 0.05 * corner_hole_score
            + 0.03 * skin_penalty
        )
        return _StructureMetrics(
            score=float(np.clip(score, 0.0, 1.0)),
            skin_ratio=skin_ratio,
            canonical_score=float(np.clip(canonical_score, 0.0, 1.0)),
            header_score=float(np.clip(header_score, 0.0, 1.0)),
            header_hole_score=float(np.clip(header_hole_score, 0.0, 1.0)),
            corner_hole_score=float(np.clip(corner_hole_score, 0.0, 1.0)),
            module_score=float(np.clip(module_score, 0.0, 1.0)),
            connector_score=float(np.clip(connector_score, 0.0, 1.0)),
            identity_score=float(np.clip(identity_score, 0.0, 1.0)),
            edge_density_score=float(np.clip(edge_density_score, 0.0, 1.0)),
            edge_grid_score=float(np.clip(edge_grid_score, 0.0, 1.0)),
            color_score=color_score,
        )

    @staticmethod
    def _header_hole_band_score(hsv: np.ndarray, edges: np.ndarray) -> float:
        h, w = edges.shape[:2]

        def band_score(y1f: float, y2f: float) -> float:
            y1 = int(round(y1f * h))
            y2 = int(round(y2f * h))
            x1 = int(round(0.06 * w))
            x2 = int(round(0.96 * w))
            band_hsv = hsv[y1:y2, x1:x2]
            band_edges = edges[y1:y2, x1:x2]
            if band_hsv.size == 0 or band_edges.size == 0:
                return 0.0

            val = band_hsv[:, :, 2]
            sat = band_hsv[:, :, 1]
            edge_support = cv.dilate(band_edges, np.ones((3, 3), np.uint8), iterations=1) > 0
            pad_like = edge_support & (val > 85) & (sat < 210)
            projection = np.mean(pad_like, axis=0).astype(np.float32)
            if projection.size == 0:
                return 0.0
            projection = cv.blur(projection.reshape(1, -1), (9, 1)).reshape(-1)
            threshold = float(np.mean(projection) + 0.55 * np.std(projection))
            segments = BoardLocalizer._count_projection_segments(projection, threshold, min_width=2)
            segment_score = float(np.clip((segments - 5) / 18.0, 0.0, 1.0))

            density = float(np.mean(pad_like))
            density_score = float(np.clip((density - 0.012) / 0.075, 0.0, 1.0))
            return float(np.clip(0.62 * segment_score + 0.38 * density_score, 0.0, 1.0))

        top = band_score(0.045, 0.215)
        bottom = band_score(0.785, 0.955)
        return float(np.clip(0.58 * max(top, bottom) + 0.42 * min(top, bottom), 0.0, 1.0))

    @staticmethod
    def _count_projection_segments(projection: np.ndarray, threshold: float, *, min_width: int) -> int:
        if projection.size == 0:
            return 0
        active = projection > threshold
        count = 0
        start: int | None = None
        for idx, is_active in enumerate(active):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                if idx - start >= min_width:
                    count += 1
                start = None
        if start is not None and active.size - start >= min_width:
            count += 1
        return count

    @staticmethod
    def _corner_hole_pattern_score(hsv: np.ndarray, edges: np.ndarray) -> float:
        h, w = edges.shape[:2]
        rois = [
            (0.015, 0.025, 0.145, 0.245),
            (0.855, 0.025, 0.985, 0.245),
            (0.015, 0.755, 0.145, 0.975),
            (0.855, 0.755, 0.985, 0.975),
        ]
        scores: list[float] = []
        for x1f, y1f, x2f, y2f in rois:
            x1 = int(round(x1f * w))
            y1 = int(round(y1f * h))
            x2 = int(round(x2f * w))
            y2 = int(round(y2f * h))
            roi_hsv = hsv[y1:y2, x1:x2]
            roi_edges = edges[y1:y2, x1:x2]
            scores.append(BoardLocalizer._corner_hole_roi_score(roi_hsv, roi_edges))
        if not scores:
            return 0.0
        scores.sort(reverse=True)
        return float(np.clip(0.52 * np.mean(scores[:2]) + 0.48 * np.mean(scores), 0.0, 1.0))

    @staticmethod
    def _corner_hole_roi_score(roi_hsv: np.ndarray, roi_edges: np.ndarray) -> float:
        if roi_hsv.size == 0 or roi_edges.size == 0:
            return 0.0
        val = roi_hsv[:, :, 2]
        sat = roi_hsv[:, :, 1]
        contrast_mask = ((val < 90) | ((val > 145) & (sat < 150))).astype(np.uint8) * 255
        edge_mask = cv.dilate(roi_edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv.bitwise_and(contrast_mask, edge_mask)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, roi_edges.shape[0] * roi_edges.shape[1]))
        best = 0.0
        for contour in contours:
            area = float(cv.contourArea(contour))
            area_ratio = area / roi_area
            if area_ratio < 0.006 or area_ratio > 0.42:
                continue
            perimeter = float(cv.arcLength(contour, True))
            if perimeter <= 1.0:
                continue
            x, y, bw, bh = cv.boundingRect(contour)
            if bw <= 2 or bh <= 2:
                continue
            aspect = bw / max(1.0, float(bh))
            aspect_score = float(np.exp(-1.4 * abs(np.log(max(1e-6, aspect)))))
            circularity = float(np.clip(4.0 * np.pi * area / max(1.0, perimeter * perimeter), 0.0, 1.0))
            size_score = float(np.exp(-abs(np.log(max(1e-6, area_ratio) / 0.095))))
            best = max(best, 0.38 * aspect_score + 0.34 * circularity + 0.28 * size_score)
        return float(np.clip(best, 0.0, 1.0))

    @staticmethod
    def _rect_shape_score(
        mask: np.ndarray,
        min_area_ratio: float,
        max_area_ratio: float,
        min_aspect: float,
        max_aspect: float,
    ) -> float:
        if mask.size == 0:
            return 0.0
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        roi_area = float(max(1, mask.shape[0] * mask.shape[1]))
        best = 0.0
        for contour in contours:
            area = float(cv.contourArea(contour))
            area_ratio = area / roi_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            x, y, bw, bh = cv.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            rectangularity = float(np.clip(area / max(1.0, float(bw * bh)), 0.0, 1.0))
            aspect = bw / max(1.0, float(bh))
            normalized_aspect = aspect if aspect >= 1.0 else 1.0 / aspect
            if normalized_aspect < min_aspect or normalized_aspect > max_aspect:
                continue
            target_mid = max(1e-6, 0.5 * (min_area_ratio + max_area_ratio))
            area_score = float(np.exp(-abs(np.log(max(1e-6, area_ratio) / target_mid))))
            aspect_mid = max(1e-6, 0.5 * (min_aspect + max_aspect))
            aspect_score = float(np.exp(-0.8 * abs(np.log(max(1e-6, normalized_aspect) / aspect_mid))))
            best = max(best, 0.44 * rectangularity + 0.34 * area_score + 0.22 * aspect_score)
        return float(np.clip(best, 0.0, 1.0))

    @staticmethod
    def _edge_grid_score(edges: np.ndarray) -> float:
        if edges.size == 0:
            return 0.0
        rows, cols = 4, 6
        h, w = edges.shape[:2]
        active = 0
        densities: list[float] = []
        for row in range(rows):
            y1 = int(round(row * h / rows))
            y2 = int(round((row + 1) * h / rows))
            for col in range(cols):
                x1 = int(round(col * w / cols))
                x2 = int(round((col + 1) * w / cols))
                cell = edges[y1:y2, x1:x2]
                density = float(np.mean(cell > 0)) if cell.size else 0.0
                densities.append(density)
                if density > 0.018:
                    active += 1
        occupancy_score = float(np.clip((active - 5) / 14.0, 0.0, 1.0))
        density_balance = float(np.clip(np.percentile(densities, 75) / max(0.015, np.percentile(densities, 25) + 0.015), 0.0, 3.0))
        balance_score = float(np.clip(density_balance / 2.2, 0.0, 1.0))
        return float(np.clip(0.72 * occupancy_score + 0.28 * balance_score, 0.0, 1.0))

    def _warp_quality_score(
        self,
        quad: np.ndarray,
        warped: np.ndarray,
        geometry_score: float,
        verify_score: float,
        objectness_score: float,
        pcb_structure_score: float,
        tightness_score: float,
    ) -> float:
        """
        Score whether the candidate warp is good enough for fixed canonical ROIs.

        This is deliberately separate from board confidence. A clutter rectangle can
        have plausible geometry, but bad edge balance, poor board fill, or weak
        canonical left/right structure should keep it from driving components.
        """
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])

        def balance_score(a: float, b: float) -> float:
            ratio = max(a, b) / max(1e-6, min(a, b))
            return float(np.exp(-1.35 * abs(np.log(ratio))))

        edge_balance = 0.5 * balance_score(top, bottom) + 0.5 * balance_score(left, right)

        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        margin_y = max(2, int(0.035 * h))
        margin_x = max(2, int(0.025 * w))
        border = np.concatenate(
            [
                hsv[:margin_y, :, :].reshape(-1, 3),
                hsv[h - margin_y :, :, :].reshape(-1, 3),
                hsv[:, :margin_x, :].reshape(-1, 3),
                hsv[:, w - margin_x :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        white_border_ratio = float(np.mean((border[:, 2] > 185) & (border[:, 1] < 70))) if border.size else 1.0
        border_fill_score = float(np.clip(1.0 - (white_border_ratio - 0.10) / 0.55, 0.0, 1.0))

        verify_quality = float(np.clip((verify_score + 0.08) / 0.58, 0.0, 1.0)) if self._references else 1.0

        quality = (
            0.18 * float(np.clip(geometry_score, 0.0, 1.0))
            + 0.19 * verify_quality
            + 0.18 * float(np.clip(objectness_score, 0.0, 1.0))
            + 0.15 * float(np.clip(pcb_structure_score, 0.0, 1.0))
            + 0.12 * float(np.clip(tightness_score, 0.0, 1.0))
            + 0.10 * border_fill_score
            + 0.08 * edge_balance
        )
        return float(np.clip(quality, 0.0, 1.0))

    def _canonical_structure_score(self, warped: np.ndarray) -> float:
        return self._canonical_region_scores(warped)[0]

    def _canonical_region_scores(self, warped: np.ndarray) -> tuple[float, float, float]:
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        gray = clahe_gray(gray)
        edges = cv.Canny(gray, self._cfg.canny_t1, self._cfg.canny_t2)
        h, w = gray.shape[:2]

        def roi(x1f: float, y1f: float, x2f: float, y2f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x1 = int(round(x1f * w))
            y1 = int(round(y1f * h))
            x2 = int(round(x2f * w))
            y2 = int(round(y2f * h))
            return gray[y1:y2, x1:x2], hsv[y1:y2, x1:x2], edges[y1:y2, x1:x2]

        esp_gray, _esp_hsv, esp_edges = roi(0.04, 0.10, 0.48, 0.88)
        esp_hsv = hsv[int(round(0.10 * h)) : int(round(0.88 * h)), int(round(0.04 * w)) : int(round(0.48 * w))]
        _conn_gray, conn_hsv, conn_edges = roi(0.66, 0.10, 0.98, 0.88)
        _usb_gray, usb_hsv, usb_edges = roi(0.74, 0.12, 0.98, 0.56)
        _jst_gray, jst_hsv, jst_edges = roi(0.70, 0.42, 0.98, 0.88)
        if esp_gray.size == 0 or conn_hsv.size == 0:
            return 0.0, 0.0, 0.0

        esp_dark_ratio = float(np.mean(esp_gray < 125))
        esp_edge_density = float(np.mean(esp_edges > 0)) if esp_edges.size else 0.0
        esp_mid_or_dark = (esp_hsv[:, :, 2] < 150) | ((esp_hsv[:, :, 2] < 205) & (esp_hsv[:, :, 1] < 115))
        esp_shape_score = self._rect_shape_score(esp_mid_or_dark, 0.08, 0.78, 0.70, 3.00)
        connector_bright_ratio = float(np.mean((conn_hsv[:, :, 2] > 145) & (conn_hsv[:, :, 1] < 115)))
        connector_edge_density = float(np.mean(conn_edges > 0)) if conn_edges.size else 0.0

        esp_score = 0.42 * np.clip((esp_dark_ratio - 0.18) / 0.42, 0.0, 1.0) + 0.34 * np.clip(
            (esp_edge_density - 0.025) / 0.12,
            0.0,
            1.0,
        ) + 0.24 * esp_shape_score

        connector_broad_score = 0.54 * np.clip((connector_bright_ratio - 0.035) / 0.16, 0.0, 1.0) + 0.30 * np.clip(
            (connector_edge_density - 0.025) / 0.12,
            0.0,
            1.0,
        )

        def connector_roi_score(region_hsv: np.ndarray, region_edges: np.ndarray, *, min_area: float, max_area: float) -> float:
            if region_hsv.size == 0 or region_edges.size == 0:
                return 0.0
            bright = (region_hsv[:, :, 2] > 132) & (region_hsv[:, :, 1] < 155)
            bright_ratio = float(np.mean(bright))
            edge_density = float(np.mean(region_edges > 0))
            shape = self._rect_shape_score(bright, min_area, max_area, 0.65, 2.80)
            return float(
                np.clip(
                    0.38 * np.clip((bright_ratio - 0.030) / 0.22, 0.0, 1.0)
                    + 0.34 * np.clip((edge_density - 0.020) / 0.13, 0.0, 1.0)
                    + 0.28 * shape,
                    0.0,
                    1.0,
                )
            )

        usb_score = connector_roi_score(usb_hsv, usb_edges, min_area=0.020, max_area=0.60)
        jst_score = connector_roi_score(jst_hsv, jst_edges, min_area=0.025, max_area=0.68)
        connector_score = float(
            np.clip(
                0.38 * connector_broad_score + 0.36 * max(usb_score, jst_score) + 0.26 * min(usb_score, jst_score),
                0.0,
                1.0,
            )
        )
        canonical = float(np.clip(0.52 * esp_score + 0.48 * connector_score, 0.0, 1.0))
        return canonical, float(np.clip(esp_score, 0.0, 1.0)), connector_score

    def _normalize_orientation(self, warped: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the remaining 180-degree ambiguity using the reference-board bank."""
        rotated = cv.rotate(warped, cv.ROTATE_180)
        original_score = self._orientation_score(warped)
        rotated_score = self._orientation_score(rotated)
        if rotated_score <= original_score:
            LOGGER.debug(
                "board orientation kept: original=%.3f rotated180=%.3f",
                original_score,
                rotated_score,
            )
            return warped, homography

        out_w, out_h = self._cfg.output_size
        rotation = _rotation_180_matrix(out_w, out_h)
        LOGGER.debug(
            "board orientation rotated180: original=%.3f rotated180=%.3f",
            original_score,
            rotated_score,
        )
        return rotated, rotation @ homography

    def _orientation_score(self, warped: np.ndarray) -> float:
        reference_score = self._verify_board(warped) if self._references else 0.0
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        edges = cv.Canny(clahe_gray(gray), self._cfg.canny_t1, self._cfg.canny_t2)
        _, w = hsv.shape[:2]
        left = hsv[:, : w // 2]
        right = hsv[:, w // 2 :]
        left_edges = edges[:, : w // 2]
        right_edges = edges[:, w // 2 :]

        def bright_connector_ratio(region: np.ndarray) -> float:
            # USB/JST plastics and metal are bright with relatively low saturation.
            mask = (region[:, :, 2] > 150) & (region[:, :, 1] < 90)
            return float(np.mean(mask))

        def dark_pcb_ratio(region: np.ndarray) -> float:
            return float(np.mean(region[:, :, 2] < 80))

        connector_bias = bright_connector_ratio(right) - bright_connector_ratio(left)
        connector_score = float(np.clip(0.5 + 1.4 * connector_bias, 0.0, 1.0))

        # Canonical convention: ESP32/BLE module on the left, USB/JST on the right.
        # The left side should usually carry more dark module area and enough edge
        # structure even when exposure shifts.
        module_bias = (dark_pcb_ratio(left) + float(np.mean(left_edges > 0))) - (
            dark_pcb_ratio(right) + float(np.mean(right_edges > 0))
        )
        module_score = float(np.clip(0.5 + 0.9 * module_bias, 0.0, 1.0))
        structure_score = self._canonical_structure_score(warped)
        return 0.50 * reference_score + 0.20 * connector_score + 0.20 * module_score + 0.10 * structure_score
