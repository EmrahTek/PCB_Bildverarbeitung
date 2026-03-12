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
    """Configuration for board detection and perspective normalization."""

    # Portrait canonical view is much more suitable for the FireBeetle board.
    output_size: tuple[int, int] = (480, 960)  # (width, height)
    canny_t1: int = 40
    canny_t2: int = 140
    min_area_ratio: float = 0.08
    approx_eps_ratio: float = 0.02
    blur_ksize: int = 5
    morph_kernel: int = 5
    aspect_ratio_min: float = 1.20  # long_side / short_side
    aspect_ratio_max: float = 6.00


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if gray.dtype != np.uint8:
        gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return gray


def _segment_intersection(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _self_intersects(pts: np.ndarray) -> bool:
    pts = pts.reshape(4, 2).astype(np.float32)
    return _segment_intersection(pts[0], pts[1], pts[2], pts[3]) or _segment_intersection(pts[1], pts[2], pts[3], pts[0])


def _is_convex_quad(pts: np.ndarray) -> bool:
    pts_i = pts.reshape(-1, 1, 2).astype(np.int32)
    return bool(cv.isContourConvex(pts_i))


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as top-left, top-right, bottom-right, bottom-left.

    This version is intentionally more robust than the classic sum/diff trick.
    The old sum/diff method can break on tall, symmetric rectangles.
    """
    pts = pts.reshape(4, 2).astype(np.float32)

    # Split into top and bottom pairs by y coordinate.
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # Fallback to a centroid-angle strategy if the simple ordering is invalid.
    if _self_intersects(ordered) or not _is_convex_quad(ordered):
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        idx = np.argsort(angles)
        cyc = pts[idx]
        start = int(np.argmin(cyc.sum(axis=1)))
        cyc = np.roll(cyc, -start, axis=0)

        # Ensure clockwise TL, TR, BR, BL.
        if cyc[1, 1] > cyc[-1, 1]:
            cyc = np.array([cyc[0], cyc[-1], cyc[-2], cyc[-3]], dtype=np.float32)
        ordered = cyc.astype(np.float32)

    return ordered


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    return top, right, bottom, left


def _valid_aspect_ratio(quad: np.ndarray, cfg: BoardWarpConfig) -> bool:
    top, right, bottom, left = _quad_side_lengths(quad)
    w = max(1.0, 0.5 * (top + bottom))
    h = max(1.0, 0.5 * (left + right))
    long_side = max(w, h)
    short_side = min(w, h)
    ratio = long_side / short_side
    return cfg.aspect_ratio_min <= ratio <= cfg.aspect_ratio_max


def _candidate_quad_from_contour(cnt: np.ndarray, cfg: BoardWarpConfig) -> np.ndarray:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, cfg.approx_eps_ratio * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    return box.reshape(4, 2).astype(np.float32)


def find_board_quad(gray: np.ndarray, cfg: BoardWarpConfig) -> np.ndarray | None:
    """Find the most plausible rectangular PCB candidate in the frame."""
    gray = _ensure_gray(gray)
    blur = cv.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)

    edges = cv.Canny(blur, cfg.canny_t1, cfg.canny_t2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (cfg.morph_kernel, cfg.morph_kernel))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    edges = cv.dilate(edges, kernel, iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    frame_area = float(h * w)
    min_area = cfg.min_area_ratio * frame_area

    best_quad: np.ndarray | None = None
    best_score = -1.0

    for cnt in contours:
        area = float(cv.contourArea(cnt))
        if area < min_area:
            continue

        quad = order_quad_points(_candidate_quad_from_contour(cnt, cfg))
        quad_area = abs(float(cv.contourArea(quad.reshape(-1, 1, 2))))
        if quad_area <= 1.0:
            continue

        if _self_intersects(quad) or not _is_convex_quad(quad):
            continue

        if not _valid_aspect_ratio(quad, cfg):
            continue

        rect_fill = area / quad_area
        score = area * rect_fill
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def warp_board(bgr: np.ndarray, cfg: BoardWarpConfig) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Detect board and warp it to a portrait canonical view.

    Returns:
        (warped_bgr, H) or (None, None) if no valid board is found.
    """
    gray = _ensure_gray(bgr)
    quad = find_board_quad(gray, cfg)
    if quad is None:
        return None, None

    out_w, out_h = cfg.output_size
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    H = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, H, (out_w, out_h))

    # Guarantee portrait output. This prevents the board from being stretched into
    # a landscape canvas, which hurt ESP32 matching in your previous runs.
    if warped.shape[1] > warped.shape[0]:
        warped = cv.rotate(warped, cv.ROTATE_90_CLOCKWISE)

    return warped, H
