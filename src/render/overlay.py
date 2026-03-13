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
    default_box_color_bgr: tuple[int, int, int] = (0, 255, 0)
    board_color_bgr: tuple[int, int, int] = (255, 180, 0)
    font: int = cv.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2
    pad_px: int = 6
    label_gap_px: int = 4
    label_bg_bgr: tuple[int, int, int] = (0, 0, 0)
    label_text_bgr: tuple[int, int, int] = (255, 255, 255)


_LABEL_COLORS = {
    "BOARD": (0, 255, 0),
    "ESP32": (0, 255, 255),
    "USB_PORT": (255, 180, 0),
    "JST_CONNECTOR": (255, 0, 255),
    "RESET_BUTTON": (0, 165, 255),
}


def _pick_color_for_frame(img: np.ndarray, bgr: tuple[int, int, int], gray_fallback: int = 255):
    if img.ndim == 2:
        return gray_fallback
    if img.ndim == 3 and img.shape[2] == 3:
        return bgr
    if img.ndim == 3 and img.shape[2] == 4:
        return (*bgr, 255)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _color_for_label(img: np.ndarray, label: str, cfg: OverlayConfig):
    return _pick_color_for_frame(img, _LABEL_COLORS.get(label, cfg.default_box_color_bgr), 255)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _place_label(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    cfg: OverlayConfig,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    text_color = _pick_color_for_frame(img, cfg.label_text_bgr, 255)
    rect_color = _pick_color_for_frame(img, cfg.label_bg_bgr, 0)

    (tw, th), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
    label_h = th + baseline + 2 * cfg.pad_px
    label_w = tw + 2 * cfg.pad_px
    frame_h, frame_w = img.shape[:2]

    x1 = int(max(0, min(frame_w - label_w, x))) if frame_w > label_w else 0

    candidate_tops: list[int] = []
    preferred_top = y - label_h
    candidate_tops.append(preferred_top)

    for k in range(1, 6):
        candidate_tops.append(preferred_top - k * (label_h + cfg.label_gap_px))

    candidate_tops.append(y + cfg.label_gap_px)
    for k in range(1, 4):
        candidate_tops.append(y + cfg.label_gap_px + k * (label_h + cfg.label_gap_px))

    placed_box: tuple[int, int, int, int] | None = None
    for top in candidate_tops:
        top = int(max(0, min(frame_h - label_h, top))) if frame_h > label_h else 0
        rect = (x1, top, x1 + label_w, top + label_h)
        if not any(_boxes_intersect(rect, other) for other in occupied):
            placed_box = rect
            break

    if placed_box is None:
        top = int(max(0, min(frame_h - label_h, preferred_top))) if frame_h > label_h else 0
        placed_box = (x1, top, x1 + label_w, top + label_h)

    rx1, ry1, rx2, ry2 = placed_box
    cv.rectangle(img, (rx1, ry1), (rx2, ry2), rect_color, thickness=-1)
    text_org = (rx1 + cfg.pad_px, ry2 - cfg.pad_px - baseline)
    cv.putText(img, text, text_org, cfg.font, cfg.font_scale, text_color, cfg.font_thickness, cv.LINE_AA)
    occupied.append(placed_box)


def _draw_counts_panel(
    img: np.ndarray,
    detections: Iterable[Detection],
    cfg: OverlayConfig,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    counts = count_by_label(detections)
    if not counts:
        return

    x = 10
    y = 60
    for label, count in counts.items():
        _place_label(img, x, y, f"{label}: {count}", cfg, occupied)
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
    detections = list(detections)
    vis = frame.copy()
    fps_color = _pick_color_for_frame(vis, cfg.label_text_bgr, 255)

    if fps is not None:
        cv.putText(vis, f"FPS: {fps:5.1f}", (10, 30), cfg.font, 0.8, fps_color, 2, cv.LINE_AA)

    _draw_board_quad(vis, board_quad, cfg)

    occupied: list[tuple[int, int, int, int]] = []
    _draw_counts_panel(vis, detections, cfg, occupied)

    for det in sorted(detections, key=lambda d: d.bbox.area(), reverse=True):
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        color = _color_for_label(vis, det.label, cfg)
        cv.rectangle(vis, (x1, y1), (x2, y2), color, thickness=cfg.thickness)
        label = f"{det.label} {det.score:.2f}" if debug and cfg.draw_scores else det.label
        _place_label(vis, x1, y1, label, cfg, occupied)

    return vis