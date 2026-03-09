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

# src/render/overlay.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2 as cv
import numpy as np

from src.utils.types import Detection
@dataclass(frozen=True)
class OverlayConfig:
    """
    Controls overlay appearance.
    Keep defaults conservative and readable.
    """
    draw_scores: bool = True

    # Box style
    thickness: int = 3  # was 2
    box_color_bgr: tuple[int, int, int] = (0, 255, 0)  # green

    # Text style
    font: int = cv.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.85  # was 0.6
    font_thickness: int = 2   # was 1
    pad_px: int = 6  # padding for text box

    # Label colors (background + text)
    label_bg_bgr: tuple[int, int, int] = (0, 0, 0)        # black
    label_text_bgr: tuple[int, int, int] = (255, 255, 255) # white

def _colors_for_image(img: np.ndarray) -> tuple[object, object]:
    """
    Pick text/box colors depending on image channel count.
    OpenCV expects different scalar formats for gray vs BGR vs BGRA.
    """
    if img.ndim == 2:
        #grayscale
        return 255,0 # text white, box black
    if img.ndim == 3 and img.shape[2] == 3:
        #BGR
        return (255,255,255), (0,0,0)
    if img.ndim == 3 and img.shape[2] == 4:
        #BGRA
        return (255,255,255,255), (0,0,0,255)
    raise ValueError(f"Unsupported image shape for overlay: {img.shape}")

def _pick_color_for_frame(vis: np.ndarray, bgr: tuple[int, int, int], gray_fallback: int = 255):
    """Return a color scalar compatible with the given image (gray vs BGR vs BGRA)."""
    if vis.ndim == 2:
        return gray_fallback
    if vis.ndim == 3 and vis.shape[2] == 3:
        return bgr
    if vis.ndim == 3 and vis.shape[2] == 4:
        return (*bgr, 255)
    raise ValueError(f"Unsupported image shape for overlay: {vis.shape}")

def _put_label(img: np.ndarray, x: int, y: int, text: str, cfg: OverlayConfig) -> None:
    """
    Draw a small filled rectangle + text at (x, y) anchor.
    """
    text_color = _pick_color_for_frame(img, cfg.label_text_bgr, gray_fallback=255)
    rect_color = _pick_color_for_frame(img, cfg.label_bg_bgr, gray_fallback=0)

    (tw, th), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)

    h, w = img.shape[:2]
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))

    box_x2 = min(w, x1 + tw + 2 * cfg.pad_px)
    box_y1 = max(0, y1 - th - baseline - 2 * cfg.pad_px)
    box_y2 = min(h, y1)

    cv.rectangle(img, (x1, box_y1), (box_x2, box_y2), rect_color, thickness=-1)
    cv.putText(
        img,
        text,
        (x1 + cfg.pad_px, box_y2 - cfg.pad_px - baseline),
        cfg.font,
        cfg.font_scale,
        text_color,
        cfg.font_thickness,
        cv.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    fps: Optional[float] = None,
    debug: bool = False,
    cfg: OverlayConfig = OverlayConfig(),
) -> np.ndarray:
    """
    Draw bounding boxes + labels onto a copy of the input frame.

    Args:
        frame: Input image (gray/BGR/BGRA).
        detections: Iterable of Detection objects.
        fps: If provided, draw FPS text in top-left.
        debug: If True, include scores in labels.
        cfg: OverlayConfig for styling.

    Returns:
        A new image with overlays drawn.
    """
    vis = frame.copy()

    # Colors compatible with gray/BGR/BGRA frames
    fps_color = _pick_color_for_frame(vis, cfg.label_text_bgr, gray_fallback=255)
    box_color = _pick_color_for_frame(vis, cfg.box_color_bgr, gray_fallback=255)

    # Draw FPS first (slightly larger and readable)
    if fps is not None:
        fps_text = f"FPS: {fps:5.1f}"
        cv.putText(
            vis,
            fps_text,
            (10, 30),
            cfg.font,
            max(cfg.font_scale, 0.8),  # ensure readable fps size
            fps_color,
            max(cfg.font_thickness, 2),
            cv.LINE_AA,
        )

    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2

        # Clamp bbox (avoid OpenCV issues with negative coords)
        h, w = vis.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        # Draw rectangle (green, thicker)
        cv.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness=cfg.thickness)

        # Build label text
        label = f"{det.label} {det.score:.2f}" if (debug and cfg.draw_scores) else det.label

        # Draw label box + text
        _put_label(vis, x1, y1, label, cfg)

    return vis

