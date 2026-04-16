from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.preprocessing.color import normalize_gray, to_gray
from src.preprocessing.filters import clahe_gray, gaussian_blur
from src.utils.types import BBox


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
    verify_gray_weight: float = 0.65
    verify_edge_weight: float = 0.35


@dataclass(frozen=True)
class BoardLocalization:
    """Structured result returned by the board localizer."""
    quad: np.ndarray
    bbox: BBox
    homography: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray
    score: float


@dataclass(frozen=True)
class _BoardCandidate:
    quad: np.ndarray
    bbox: BBox
    geometry_score: float
    verify_score: float
    score: float
    homography: np.ndarray
    h_inv: np.ndarray
    warped: np.ndarray


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

    def localize(self, frame: np.ndarray, hint_bbox: BBox | None = None) -> BoardLocalization | None:
        search_boxes = [BBox(0, 0, frame.shape[1], frame.shape[0])]
        min_score = self._cfg.min_score
        if hint_bbox is not None:
            search_boxes.insert(0, expand_bbox(hint_bbox, frame.shape, self._cfg.search_expansion))
            min_score = self._cfg.min_tracked_score

        best: _BoardCandidate | None = None
        for search_box in search_boxes:
            candidate = self._find_best_candidate(frame, search_box)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
            if candidate.score >= min_score:
                break

        if best is None or best.score < min_score:
            return None

        return BoardLocalization(
            quad=best.quad,
            bbox=best.bbox,
            homography=best.homography,
            h_inv=best.h_inv,
            warped=best.warped,
            score=best.score,
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
            verify_score = self._verify_board(warped)
            score = 0.75 * geometry_score + 0.25 * max(0.0, verify_score) if self._references else geometry_score
            candidate = _BoardCandidate(
                quad=quad,
                bbox=quad_to_bbox(quad, frame.shape),
                geometry_score=geometry_score,
                verify_score=verify_score,
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
        gray = clahe_gray(normalize_gray(to_gray(resized)))
        gray = gaussian_blur(gray, 3)
        edges = cv.Canny(gray, self._cfg.canny_t1, self._cfg.canny_t2)
        return gray, edges

    def _verify_board(self, warped: np.ndarray) -> float:
        if not self._references:
            return 1.0

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

    def _normalize_orientation(self, warped: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the remaining 180-degree ambiguity using the reference-board bank."""
        if not self._references:
            return warped, homography

        rotated = cv.rotate(warped, cv.ROTATE_180)
        original_score = self._verify_board(warped)
        rotated_score = self._verify_board(rotated)
        if rotated_score <= original_score:
            return warped, homography

        out_w, out_h = self._cfg.output_size
        rotation = _rotation_180_matrix(out_w, out_h)
        return rotated, rotation @ homography
