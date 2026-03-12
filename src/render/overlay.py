# Boxes/Labels/Counts zeichnen
# Visualisierung: Bounding Boxes, Labels, Score, Count-Panel; Debug-Overlays (Board-Ecken).

"""
overlay.py

This module renders visualization overlays onto frames:
- bounding boxes and labels for detections
- confidence scores
- a counts panel showing how many instances per label were found
- optional debug overlay for board detection (corners/contours)

Inputs:
- Original or warped frame (NumPy array)
- list[Detection]
- counts dictionary (label -> int)
- optional debug_info (board corners, etc.)

Outputs:
- Annotated frame (NumPy array)

Zu implementierende Funktionen

    draw_detections(frame, detections) -> frame

    draw_counts_panel(frame, counts) -> frame

    draw_board_debug(frame, debug_info) -> frame

    put_fps(frame, fps_value) -> frame (oder via fps.py)





OpenCV drawing (rectangle, putText):
https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2 as cv
import numpy as np

from src.detection_logic.postprocess import count_by_label
from src.utils.types import Detection


@dataclass(frozen=True)
class OverlayConfig:
    draw_scores: bool = True
    thickness: int = 2
    box_color_bgr: tuple[int, int, int] = (0, 255, 0)
    board_color_bgr: tuple[int, int, int] = (255, 180, 0)
    font: int = cv.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2
    pad_px: int = 6
    label_bg_bgr: tuple[int, int, int] = (0, 0, 0)
    label_text_bgr: tuple[int, int, int] = (255, 255, 255)


def _pick_color_for_frame(img: np.ndarray, bgr: tuple[int, int, int], gray_fallback: int = 255):
    if img.ndim == 2:
        return gray_fallback
    if img.ndim == 3 and img.shape[2] == 3:
        return bgr
    if img.ndim == 3 and img.shape[2] == 4:
        return (*bgr, 255)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _put_label(img: np.ndarray, x: int, y: int, text: str, cfg: OverlayConfig) -> None:
    text_color = _pick_color_for_frame(img, cfg.label_text_bgr, 255)
    rect_color = _pick_color_for_frame(img, cfg.label_bg_bgr, 0)
    (tw, th), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
    x1 = max(0, x)
    y1 = max(0, y)
    box_y1 = max(0, y1 - th - baseline - 2 * cfg.pad_px)
    box_y2 = y1
    box_x2 = min(img.shape[1], x1 + tw + 2 * cfg.pad_px)
    cv.rectangle(img, (x1, box_y1), (box_x2, box_y2), rect_color, thickness=-1)
    cv.putText(img, text, (x1 + cfg.pad_px, box_y2 - cfg.pad_px - baseline), cfg.font, cfg.font_scale, text_color, cfg.font_thickness, cv.LINE_AA)


def _draw_counts_panel(img: np.ndarray, detections: Iterable[Detection], cfg: OverlayConfig) -> None:
    counts = count_by_label(detections)
    if not counts:
        return
    panel_lines = [f"{label}: {count}" for label, count in counts.items()]
    x = 10
    y = 60
    for line in panel_lines:
        _put_label(img, x, y, line, cfg)
        y += 28


def _draw_board_quad(img: np.ndarray, board_quad: np.ndarray | None, cfg: OverlayConfig) -> None:
    if board_quad is None:
        return
    quad = board_quad.reshape(-1, 2).astype(int)
    color = _pick_color_for_frame(img, cfg.board_color_bgr, 255)
    cv.polylines(img, [quad.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    fps: Optional[float] = None,
    debug: bool = False,
    board_quad: np.ndarray | None = None,
    cfg: OverlayConfig = OverlayConfig(),
) -> np.ndarray:
    """Draw detections, FPS, counts, and optional board quadrilateral."""
    detections = list(detections)
    vis = frame.copy()
    box_color = _pick_color_for_frame(vis, cfg.box_color_bgr, 255)
    fps_color = _pick_color_for_frame(vis, cfg.label_text_bgr, 255)

    if fps is not None:
        cv.putText(vis, f"FPS: {fps:5.1f}", (10, 30), cfg.font, 0.8, fps_color, 2, cv.LINE_AA)

    _draw_board_quad(vis, board_quad, cfg)

    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        cv.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness=cfg.thickness)
        label = f"{det.label} {det.score:.2f}" if debug and cfg.draw_scores else det.label
        _put_label(vis, x1, y1, label, cfg)

    _draw_counts_panel(vis, detections, cfg)
    return vis
