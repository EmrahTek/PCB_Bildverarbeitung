# PCB finden (Konturen), Ecken ordnen, Homographie berechnen, perspektivisch auf Standardansicht warpen.


"""
geometry.py

This module implements board localization and perspective normalization.
The idea is to find the PCB in the camera frame, estimate a homography, and warp
the view into a canonical "top-down" board coordinate system.

This step greatly improves detection robustness by stabilizing scale and orientation
across frames.

Inputs:
- BGR frame (NumPy array)
- Configuration parameters (thresholds, target output size)

Outputs:
- warped_board: normalized board image (NumPy array)
- debug_info: optional metadata for visualization (corners, contour, homography)


Zu implementierende Funktionen

    find_board_contour(gray_or_edges) -> contour | None

    approx_quad(contour) -> np.ndarray[(4,2)] | None

    order_points(pts4) -> pts4_ordered

    compute_homography(src_pts, dst_pts) -> H

    warp_perspective(frame, H, out_size=(W,H)) -> warped

    detect_and_warp_board(frame, config) -> (warped, debug_info)

Contours:
https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

findContours:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

approxPolyDP:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

Homography / warpPerspective:
https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html    

"""
from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class BoardWarpConfig:
    """Configuration for board localisation and perspective normalization."""
    output_size: tuple[int, int] = (900, 460)
    blur_ksize: int = 5
    canny_t1: int = 40
    canny_t2: int = 120
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.35
    approx_eps_ratio: float = 0.02
    expected_aspect_ratio: float = 1.95
    min_aspect_ratio: float = 1.45
    max_aspect_ratio: float = 2.55
    min_rectangularity: float = 0.55
    morph_kernel: int = 5
    border_margin: int = 6


@dataclass(frozen=True)
class BoardDetectionResult:
    quad: np.ndarray
    homography: np.ndarray
    warped: np.ndarray
    score: float


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    width = max(1e-6, 0.5 * (top + bottom))
    height = max(1e-6, 0.5 * (left + right))
    ar = width / height
    return ar if ar >= 1.0 else 1.0 / ar


def _border_touch_count(
    quad: np.ndarray,
    frame_shape: tuple[int, int],
    margin: int,
) -> int:
    h, w = frame_shape[:2]
    xs = quad[:, 0]
    ys = quad[:, 1]

    count = 0
    if float(xs.min()) <= margin:
        count += 1
    if float(ys.min()) <= margin:
        count += 1
    if float(xs.max()) >= (w - 1 - margin):
        count += 1
    if float(ys.max()) >= (h - 1 - margin):
        count += 1
    return count


def _size_score(area_ratio: float) -> float:
    """
    Prefer medium-size board candidates.
    Very tiny or very large rectangles should score poorly.
    """
    if area_ratio < 0.01:
        return 0.0
    if area_ratio <= 0.08:
        return area_ratio / 0.08
    if area_ratio <= 0.25:
        return 1.0
    if area_ratio <= 0.35:
        return 1.0 - (area_ratio - 0.25) / 0.10
    return 0.0


def _candidate_score(
    *,
    contour_area: float,
    frame_area: float,
    quad: np.ndarray,
    cfg: BoardWarpConfig,
) -> float:
    xs = quad[:, 0]
    ys = quad[:, 1]
    box_w = float(xs.max() - xs.min())
    box_h = float(ys.max() - ys.min())
    box_area = max(1.0, box_w * box_h)

    rectangularity = float(contour_area) / box_area
    rectangularity = max(0.0, min(1.0, rectangularity))

    ar = _quad_aspect_ratio(quad)
    target = max(1.0, float(cfg.expected_aspect_ratio))
    aspect_penalty = abs(np.log(ar / target))
    aspect_score = float(np.exp(-2.8 * aspect_penalty))

    area_ratio = float(contour_area / frame_area)
    size_score = _size_score(area_ratio)

    return (
        0.50 * aspect_score +
        0.30 * rectangularity +
        0.20 * size_score
    )


def _quads_from_contours(
    contours: list[np.ndarray],
    frame_area: float,
    cfg: BoardWarpConfig,
    frame_shape: tuple[int, int],
) -> list[tuple[np.ndarray, float]]:
    candidates: list[tuple[np.ndarray, float]] = []

    for cnt in contours:
        area = float(cv.contourArea(cnt))
        area_ratio = area / frame_area

        if area < cfg.min_area_ratio * frame_area:
            continue
        if area_ratio > cfg.max_area_ratio:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, cfg.approx_eps_ratio * peri, True)

        if len(approx) == 4:
            quad = order_quad_points(approx)
        else:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            quad = order_quad_points(box)

        ar = _quad_aspect_ratio(quad)
        if ar < cfg.min_aspect_ratio or ar > cfg.max_aspect_ratio:
            continue

        xs = quad[:, 0]
        ys = quad[:, 1]
        box_area = max(1.0, float((xs.max() - xs.min()) * (ys.max() - ys.min())))
        rectangularity = area / box_area
        if rectangularity < cfg.min_rectangularity:
            continue

        border_touch = _border_touch_count(quad, frame_shape, cfg.border_margin)
        if border_touch >= 2:
            continue

        score = _candidate_score(
            contour_area=area,
            frame_area=frame_area,
            quad=quad,
            cfg=cfg,
        )
        candidates.append((quad, score))

    return candidates


def find_board_quad(gray: np.ndarray, cfg: BoardWarpConfig) -> tuple[np.ndarray, float] | None:
    """Find the best board quad using edges + dark-object segmentation."""
    if gray.ndim != 2:
        raise ValueError("find_board_quad expects a grayscale image.")

    blur_k = max(3, int(cfg.blur_ksize) | 1)
    gray_blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
    h, w = gray_blur.shape[:2]
    frame_area = float(h * w)

    candidates: list[tuple[np.ndarray, float]] = []

    # Path 1: edges
    edges = cv.Canny(gray_blur, cfg.canny_t1, cfg.canny_t2)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours_e, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates.extend(_quads_from_contours(contours_e, frame_area, cfg, gray.shape))

    # Path 2: dark object mask
    _, mask = cv.threshold(gray_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    k = max(3, int(cfg.morph_kernel) | 1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (k, k))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    contours_m, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    candidates.extend(_quads_from_contours(contours_m, frame_area, cfg, gray.shape))

    if not candidates:
        return None

    best_quad, best_score = max(candidates, key=lambda item: item[1])
    return best_quad.astype(np.float32), float(best_score)


def detect_and_warp_board(bgr: np.ndarray, cfg: BoardWarpConfig) -> BoardDetectionResult | None:
    """Detect the board quad and warp to a canonical output view."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    found = find_board_quad(gray, cfg)
    if found is None:
        return None

    quad, score = found

    out_w, out_h = cfg.output_size
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, H, (out_w, out_h))

    return BoardDetectionResult(
        quad=quad,
        homography=H,
        warped=warped,
        score=float(score),
    )


def warp_board(bgr: np.ndarray, cfg: BoardWarpConfig) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Backward-compatible wrapper used by older code."""
    result = detect_and_warp_board(bgr, cfg)
    if result is None:
        return None, None
    return result.warped, result.homography