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
    output_size: tuple[int, int] = (800, 600)
    canny_t1: int = 50
    canny_t2: int = 150
    min_area_ratio: float = 0.10
    approx_eps_ratio: float = 0.02
    blur_ksize: int = 5
    morph_ksize: int = 5


@dataclass(frozen=True)
class BoardLocalization:
    warped: np.ndarray
    h: np.ndarray
    h_inv: np.ndarray
    quad: np.ndarray
    reused: bool = False


class BoardLocalizer:
    """Cache board localization and only redetect every N frames."""

    def __init__(self, cfg: BoardWarpConfig, *, enabled: bool = True, redetect_interval: int = 5) -> None:
        self._cfg = cfg
        self._enabled = enabled
        self._redetect_interval = max(1, redetect_interval)
        self._cached: BoardLocalization | None = None

    def localize(self, frame: np.ndarray, frame_id: int) -> BoardLocalization | None:
        if not self._enabled:
            return None
        if self._cached is not None and frame_id % self._redetect_interval != 0:
            warped = cv.warpPerspective(frame, self._cached.h, self._cfg.output_size)
            return BoardLocalization(
                warped=warped,
                h=self._cached.h,
                h_inv=self._cached.h_inv,
                quad=self._cached.quad,
                reused=True,
            )

        warped, h, quad = warp_board(frame, self._cfg)
        if warped is None or h is None or quad is None:
            return None
        h_inv = np.linalg.inv(h)
        self._cached = BoardLocalization(warped=warped, h=h, h_inv=h_inv, quad=quad, reused=False)
        return self._cached


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four contour points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_board_quad(gray: np.ndarray, cfg: BoardWarpConfig) -> np.ndarray | None:
    """Find the largest four-corner contour that looks like the PCB board."""
    k = max(3, int(cfg.blur_ksize))
    if k % 2 == 0:
        k += 1
    blur = cv.GaussianBlur(gray, (k, k), 0)
    edges = cv.Canny(blur, cfg.canny_t1, cfg.canny_t2)
    kernel = np.ones((max(3, cfg.morph_ksize), max(3, cfg.morph_ksize)), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = cfg.min_area_ratio * float(h * w)
    best: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, cfg.approx_eps_ratio * perimeter, True)
        if len(approx) != 4:
            continue
        if area > best_area:
            best = approx
            best_area = area

    if best is None:
        return None
    return order_quad_points(best)


def warp_board(bgr: np.ndarray, cfg: BoardWarpConfig) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Detect the PCB board and warp it into a canonical top view."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    quad = find_board_quad(gray, cfg)
    if quad is None:
        return None, None, None

    out_w, out_h = cfg.output_size
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    h = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, h, (out_w, out_h))
    return warped, h, quad
