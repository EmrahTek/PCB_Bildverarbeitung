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


@dataclass(frozen=True)
class _BoardCandidate:
    quad: np.ndarray
    bbox: BBox
    geometry_score: float
    verify_score: float
    objectness_score: float
    warp_quality_score: float
    pcb_structure_score: float
    tightness_score: float
    score: float
    homography: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray


@dataclass(frozen=True)
class _StructureMetrics:
    score: float
    skin_ratio: float
    canonical_score: float
    header_score: float
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
        for search_box in search_boxes:
            candidate = self._find_best_candidate(frame, search_box)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
            if candidate.score >= min_score:
                break

        if best is None:
            LOGGER.debug(
                "board rejected: no candidate hint=%s include_full_frame=%s",
                hint_bbox is not None,
                include_full_frame,
            )
            return None

        if best.score < min_score or best.warp_quality_score < min_warp_quality:
            reasons: list[str] = []
            if best.score < min_score:
                reasons.append("low_score")
            if best.warp_quality_score < min_warp_quality:
                reasons.append("low_warp_quality")
            LOGGER.debug(
                "board rejected: reason=%s score=%.3f min_score=%.3f warp_quality=%.3f "
                "min_warp_quality=%.3f geometry=%.3f verify=%.3f objectness=%.3f "
                "structure=%.3f tightness=%.3f bbox=%s",
                "+".join(reasons),
                best.score,
                min_score,
                best.warp_quality_score,
                min_warp_quality,
                best.geometry_score,
                best.verify_score,
                best.objectness_score,
                best.pcb_structure_score,
                best.tightness_score,
                best.bbox,
            )
            return None

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
        )

    def _find_best_candidate(self, frame: np.ndarray, search_box: BBox) -> _BoardCandidate | None:
        crop = frame[search_box.y1:search_box.y2, search_box.x1:search_box.x2]
        if crop.size == 0:
            return None

        gray = normalize_gray(to_gray(crop))
        gray = clahe_gray(gray)
        gray = gaussian_blur(gray, self._cfg.blur_ksize)

        candidates = self._candidate_quads(gray, search_box, frame.shape)
        if not candidates:
            return None

        out_w, out_h = self._cfg.output_size
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )

        best: _BoardCandidate | None = None
        for quad, geometry_score in candidates:
            homography = cv.getPerspectiveTransform(quad, dst)
            warped = cv.warpPerspective(frame, homography, (out_w, out_h))
            warped, homography = self._normalize_orientation(warped, homography)
            warped, homography, quad, tightness_score = self._refine_warp_tightness(frame, warped, homography, quad)
            verify_score = self._verify_board(warped)
            objectness_score = self._board_objectness_score(warped)
            if objectness_score < self._cfg.min_objectness_score:
                LOGGER.debug(
                    "board candidate rejected: reason=low_objectness objectness=%.3f min=%.3f geometry=%.3f verify=%.3f",
                    objectness_score,
                    self._cfg.min_objectness_score,
                    geometry_score,
                    verify_score,
                )
                continue
            structure = self._pcb_structure_metrics(warped)
            if structure.skin_ratio > self._cfg.max_skin_ratio:
                LOGGER.debug(
                    "board candidate rejected: reason=skin_like_region skin=%.3f max=%.3f geometry=%.3f verify=%.3f objectness=%.3f",
                    structure.skin_ratio,
                    self._cfg.max_skin_ratio,
                    geometry_score,
                    verify_score,
                    objectness_score,
                )
                continue
            if structure.canonical_score < self._cfg.min_canonical_structure_score:
                LOGGER.debug(
                    "board candidate rejected: reason=low_canonical_structure canonical=%.3f min=%.3f "
                    "structure=%.3f header=%.3f grid=%.3f geometry=%.3f verify=%.3f",
                    structure.canonical_score,
                    self._cfg.min_canonical_structure_score,
                    structure.score,
                    structure.header_score,
                    structure.edge_grid_score,
                    geometry_score,
                    verify_score,
                )
                continue
            if structure.edge_grid_score < self._cfg.min_edge_grid_score:
                LOGGER.debug(
                    "board candidate rejected: reason=low_edge_distribution grid=%.3f min=%.3f "
                    "structure=%.3f canonical=%.3f header=%.3f geometry=%.3f verify=%.3f",
                    structure.edge_grid_score,
                    self._cfg.min_edge_grid_score,
                    structure.score,
                    structure.canonical_score,
                    structure.header_score,
                    geometry_score,
                    verify_score,
                )
                continue
            if structure.score < self._cfg.min_pcb_structure_score:
                LOGGER.debug(
                    "board candidate rejected: reason=low_pcb_structure structure=%.3f min=%.3f "
                    "canonical=%.3f header=%.3f grid=%.3f color=%.3f geometry=%.3f verify=%.3f objectness=%.3f",
                    structure.score,
                    self._cfg.min_pcb_structure_score,
                    structure.canonical_score,
                    structure.header_score,
                    structure.edge_grid_score,
                    structure.color_score,
                    geometry_score,
                    verify_score,
                    objectness_score,
                )
                continue
            if tightness_score < self._cfg.min_tightness_score:
                LOGGER.debug(
                    "board candidate rejected: reason=loose_warp tightness=%.3f min=%.3f structure=%.3f geometry=%.3f verify=%.3f",
                    tightness_score,
                    self._cfg.min_tightness_score,
                    structure.score,
                    geometry_score,
                    verify_score,
                )
                continue
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
            score = 0.82 * base_score + 0.18 * warp_quality_score
            candidate = _BoardCandidate(
                quad=quad,
                bbox=quad_to_bbox(quad, frame.shape),
                geometry_score=geometry_score,
                verify_score=verify_score,
                objectness_score=objectness_score,
                warp_quality_score=warp_quality_score,
                pcb_structure_score=structure.score,
                tightness_score=tightness_score,
                score=score,
                homography=homography,
                h_inv=np.linalg.inv(homography),
                warped=warped,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

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

        pad_x = int(round(0.035 * bw))
        pad_y = int(round(0.045 * bh))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w - 1, x + bw + pad_x)
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
            "board warp refined: tightness %.3f -> %.3f coverage=(%.2f, %.2f) aspect_score=%.3f",
            tightness,
            refined_tightness,
            coverage_x,
            coverage_y,
            aspect_score,
        )
        return refined_warped, refined_h, source_pts.astype(np.float32), max(tightness, refined_tightness)

    def _foreground_bbox_in_warp(self, warped: np.ndarray) -> tuple[int, int, int, int, float] | None:
        hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        gray = normalize_gray(to_gray(warped))
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        skin = self._skin_mask(hue, sat, val)

        dark_board = gray < 155
        saturated_pcb = (sat > 45) & (val < 220)
        mask = (dark_board | saturated_pcb) & ~skin
        mask_u8 = (mask.astype(np.uint8)) * 255
        kernel = np.ones((7, 7), np.uint8)
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, kernel, iterations=2)
        mask_u8 = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        min_area = 0.025 * h * w
        kept = [contour for contour in contours if cv.contourArea(contour) >= min_area]
        if not kept:
            return None

        points = np.vstack(kept)
        x, y, bw, bh = cv.boundingRect(points)
        fill_ratio = float(np.mean(mask_u8[y : y + bh, x : x + bw] > 0)) if bw > 0 and bh > 0 else 0.0
        return int(x), int(y), int(bw), int(bh), fill_ratio

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

        top_score = header_band_score(top_band)
        bottom_score = header_band_score(bottom_band)
        header_score = 0.55 * max(top_score, bottom_score) + 0.45 * min(top_score, bottom_score)

        canonical_score = self._canonical_structure_score(warped)
        skin_penalty = float(np.clip(1.0 - skin_ratio / max(1e-6, self._cfg.max_skin_ratio), 0.0, 1.0))
        score = (
            0.28 * canonical_score
            + 0.25 * header_score
            + 0.18 * edge_density_score
            + 0.14 * edge_grid_score
            + 0.08 * color_score
            + 0.07 * skin_penalty
        )
        return _StructureMetrics(
            score=float(np.clip(score, 0.0, 1.0)),
            skin_ratio=skin_ratio,
            canonical_score=float(np.clip(canonical_score, 0.0, 1.0)),
            header_score=float(np.clip(header_score, 0.0, 1.0)),
            edge_density_score=float(np.clip(edge_density_score, 0.0, 1.0)),
            edge_grid_score=float(np.clip(edge_grid_score, 0.0, 1.0)),
            color_score=color_score,
        )

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
        _conn_gray, conn_hsv, conn_edges = roi(0.66, 0.10, 0.98, 0.88)
        if esp_gray.size == 0 or conn_hsv.size == 0:
            return 0.0

        esp_dark_ratio = float(np.mean(esp_gray < 125))
        esp_edge_density = float(np.mean(esp_edges > 0)) if esp_edges.size else 0.0
        connector_bright_ratio = float(np.mean((conn_hsv[:, :, 2] > 145) & (conn_hsv[:, :, 1] < 115)))
        connector_edge_density = float(np.mean(conn_edges > 0)) if conn_edges.size else 0.0

        esp_score = 0.55 * np.clip((esp_dark_ratio - 0.18) / 0.42, 0.0, 1.0) + 0.45 * np.clip(
            (esp_edge_density - 0.025) / 0.12,
            0.0,
            1.0,
        )
        connector_score = 0.65 * np.clip((connector_bright_ratio - 0.035) / 0.16, 0.0, 1.0) + 0.35 * np.clip(
            (connector_edge_density - 0.025) / 0.12,
            0.0,
            1.0,
        )
        return float(np.clip(0.52 * esp_score + 0.48 * connector_score, 0.0, 1.0))

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
