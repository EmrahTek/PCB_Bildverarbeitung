from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from src.detection_logic.postprocess import count_by_label
from src.utils.types import Detection


@dataclass(frozen=True)
class OverlayConfig:
    """Visual settings for the runtime overlay."""
    draw_scores: bool = True
    show_counts: bool = True
    thickness: int = 2
    font: int = cv.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.46
    font_thickness: int = 1
    panel_x: int = 12
    panel_y: int = 28
    panel_row_gap: int = 6


_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "BOARD": (0, 255, 0),
    "ESP32": (0, 255, 255),
    "USB_PORT": (255, 255, 0),
    "JST_CONNECTOR": (255, 0, 255),
    "RESET_BUTTON": (0, 165, 255),
}

_SHORT_LABELS: dict[str, str] = {
    "BOARD": "BRD",
    "ESP32": "ESP",
    "USB_PORT": "USB",
    "JST_CONNECTOR": "JST",
    "RESET_BUTTON": "RST",
}


def _pick_color_for_frame(image: np.ndarray, bgr: tuple[int, int, int], gray_fallback: int = 255):
    if image.ndim == 2:
        return gray_fallback
    if image.ndim == 3 and image.shape[2] == 3:
        return bgr
    if image.ndim == 3 and image.shape[2] == 4:
        return (*bgr, 255)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _text_color_for_bgr(image: np.ndarray, bgr: tuple[int, int, int]):
    luminance = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    text_bgr = (0, 0, 0) if luminance > 170 else (255, 255, 255)
    return _pick_color_for_frame(image, text_bgr, gray_fallback=0 if luminance > 170 else 255)


def _draw_label_box(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    background_bgr: tuple[int, int, int],
    cfg: OverlayConfig,
) -> None:
    bg = _pick_color_for_frame(image, background_bgr, 255)
    fg = _text_color_for_bgr(image, background_bgr)
    (text_w, text_h), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
    label_h = text_h + baseline + 8
    label_w = text_w + 8

    max_x1 = max(0, image.shape[1] - label_w)
    x1 = min(max(0, x), max_x1)
    y2 = min(image.shape[0] - 1, max(label_h, y))
    y1 = max(0, y2 - label_h)
    x2 = min(image.shape[1], x1 + label_w)

    cv.rectangle(image, (x1, y1), (x2, y2), bg, thickness=-1)
    cv.putText(
        image,
        text,
        (x1 + 4, y2 - baseline - 4),
        cfg.font,
        cfg.font_scale,
        fg,
        cfg.font_thickness,
        cv.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    fps: float | None = None,
    debug: bool = False,
    cfg: OverlayConfig = OverlayConfig(),
) -> np.ndarray:
    """Draw detections, counts, and optional FPS text on top of the frame."""
    vis = frame.copy()
    detections_list = list(detections)

    for detection in detections_list:
        color = _LABEL_COLORS.get(detection.label, (0, 255, 0))
        draw_color = _pick_color_for_frame(vis, color)
        x1 = max(0, detection.bbox.x1)
        y1 = max(0, detection.bbox.y1)
        x2 = min(vis.shape[1] - 1, detection.bbox.x2)
        y2 = min(vis.shape[0] - 1, detection.bbox.y2)
        cv.rectangle(vis, (x1, y1), (x2, y2), draw_color, thickness=cfg.thickness)

        text = _SHORT_LABELS.get(detection.label, detection.label)
        if debug and cfg.draw_scores:
            text = f"{text} {detection.score:.2f}"
        label_y = max(0, y1 - 2)
        if detection.label == "JST_CONNECTOR":
            (_text_w, text_h), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
            label_y = y2 + text_h + baseline + 8
        _draw_label_box(vis, x1, label_y, text, color, cfg)

    if cfg.show_counts:
        counts = count_by_label(detections_list)
        ordered_labels = ["BOARD", "ESP32", "USB_PORT", "JST_CONNECTOR", "RESET_BUTTON"]
        row_y = cfg.panel_y
        for label in ordered_labels:
            if label not in counts:
                continue
            text = f"{_SHORT_LABELS.get(label, label)}: {counts[label]}"
            color = _LABEL_COLORS.get(label, (0, 255, 0))
            _draw_label_box(vis, cfg.panel_x, row_y, text, color, cfg)
            (text_w, text_h), baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
            row_y += text_h + baseline + cfg.panel_row_gap + 8

    if fps is not None:
        text = f"FPS {fps:4.1f}"
        (text_w, _text_h), _baseline = cv.getTextSize(text, cfg.font, cfg.font_scale, cfg.font_thickness)
        _draw_label_box(vis, vis.shape[1] - text_w - 20, 24, text, (40, 40, 40), cfg)

    return vis
