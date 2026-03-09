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

# src/preprocessing/geometry.py
from __future__ import annotations

from dataclasses import dataclass
import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class BoardWarpConfig:
    """
    Configuration for PCB board detection + homography warp.
    """
    output_size: tuple[int, int] = (800, 600)  # (W, H) canonical board view
    canny_t1: int = 50
    canny_t2: int = 150
    min_area_ratio: float = 0.10  # quad must cover at least this fraction of frame area
    approx_eps_ratio: float = 0.02  # contour approximation


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_board_quad(gray: np.ndarray, cfg: BoardWarpConfig) -> np.ndarray | None:
    """
    Find the largest 4-point contour candidate for the board.

    Returns:
        (4,2) float32 points or None.
    """
    edges = cv.Canny(gray, cfg.canny_t1, cfg.canny_t2)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    frame_area = float(h * w)

    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < cfg.min_area_ratio * frame_area:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, cfg.approx_eps_ratio * peri, True)

        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None

    pts = order_quad_points(best)
    return pts


def warp_board(bgr: np.ndarray, cfg: BoardWarpConfig) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Detect board quad and warp to canonical view.

    Returns:
        (warped_bgr, H) or (None, None) if not found.
    """
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    quad = find_board_quad(gray, cfg)
    if quad is None:
        return None, None

    out_w, out_h = cfg.output_size
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

    H = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, H, (out_w, out_h))
    return warped, H