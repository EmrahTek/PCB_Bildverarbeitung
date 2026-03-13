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
    show_counts: bool = True
    thickness: int = 2

    font: int = cv.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.46 # 0.42
    font_thickness: int = 1

    small_font_scale: float = 0.40 # 0.36
    small_font_thickness: int = 1

    count_font_scale: float = 0.44 # 0.42
    count_font_thickness: int = 1

    label_pad_x: int = 4
    label_pad_y: int = 3
    label_gap: int = 4

    count_x: int = 12
    count_y: int = 36
    count_row_gap: int = 6


_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "BOARD": (0, 255, 0),           # green
    "ESP32": (0, 255, 255),         # yellow
    "USB_PORT": (255, 255, 0),      # cyan
    "JST_CONNECTOR": (255, 0, 255), # magenta
    "RESET_BUTTON": (0, 165, 255),  # orange
}

_SHORT_LABELS: dict[str, str] = {
    "BOARD": "BRD",
    "ESP32": "ESP",
    "USB_PORT": "USB",
    "JST_CONNECTOR": "JST",
    "RESET_BUTTON": "RST",
}


def _pick_color_for_frame(img: np.ndarray, bgr: tuple[int, int, int], gray_fallback: int = 255):
    if img.ndim == 2:
        return gray_fallback
    if img.ndim == 3 and img.shape[2] == 3:
        return bgr
    if img.ndim == 3 and img.shape[2] == 4:
        return (*bgr, 255)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _label_color(img: np.ndarray, label: str):
    return _pick_color_for_frame(img, _LABEL_COLORS.get(label, (0, 255, 0)), 255)


def _text_color_for_bgr(img: np.ndarray, bgr: tuple[int, int, int]):
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    text_bgr = (0, 0, 0) if luminance > 170 else (255, 255, 255)
    return _pick_color_for_frame(img, text_bgr, 0 if luminance > 170 else 255)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _clamp_rect_to_frame(
    rect: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    if w >= frame_w:
        x1, x2 = 0, frame_w
    else:
        x1 = max(0, min(frame_w - w, x1))
        x2 = x1 + w

    if h >= frame_h:
        y1, y2 = 0, frame_h
    else:
        y1 = max(0, min(frame_h - h, y1))
        y2 = y1 + h

    return x1, y1, x2, y2


def _named_candidate_rects(
    bbox: tuple[int, int, int, int],
    label_w: int,
    label_h: int,
    frame_w: int,
    frame_h: int,
    gap: int,
) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox

    rects = {
        "above_left": (x1, y1 - label_h - gap, x1 + label_w, y1 - gap),
        "above_right": (x2 - label_w, y1 - label_h - gap, x2, y1 - gap),
        "below_left": (x1, y2 + gap, x1 + label_w, y2 + gap + label_h),
        "below_right": (x2 - label_w, y2 + gap, x2, y2 + gap + label_h),
        "right_top": (x2 + gap, y1, x2 + gap + label_w, y1 + label_h),
        "right_bottom": (x2 + gap, y2 - label_h, x2 + gap + label_w, y2),
        "left_top": (x1 - gap - label_w, y1, x1 - gap, y1 + label_h),
        "left_bottom": (x1 - gap - label_w, y2 - label_h, x1 - gap, y2),
    }

    return {k: _clamp_rect_to_frame(v, frame_w, frame_h) for k, v in rects.items()}


def _ordered_candidate_names(det: Detection) -> list[str]:
    w = det.bbox.x2 - det.bbox.x1
    h = det.bbox.y2 - det.bbox.y1
    is_tall = h > w

    if det.label == "BOARD":
        if is_tall:
            return ["right_top", "left_top", "right_bottom", "left_bottom", "above_left", "below_left"]
        return ["above_left", "above_right", "below_left", "below_right", "right_top", "left_top"]

    if det.label == "USB_PORT":
        if is_tall:
            return ["right_top", "right_bottom", "above_right", "below_right", "left_top", "left_bottom"]
        return ["right_top", "above_right", "below_right", "right_bottom", "above_left", "below_left"]

    if det.label == "JST_CONNECTOR":
        if is_tall:
            return ["right_bottom", "right_top", "below_right", "above_right", "left_bottom", "left_top"]
        return ["below_right", "right_bottom", "above_right", "right_top", "below_left", "above_left"]

    if det.label == "RESET_BUTTON":
        if is_tall:
            return ["left_top", "right_top", "above_left", "above_right", "left_bottom", "right_bottom"]
        return ["above_right", "above_left", "right_top", "left_top", "below_right", "below_left"]

    # ESP32 and fallback
    if is_tall:
        return ["left_top", "right_top", "left_bottom", "right_bottom", "above_left", "below_left"]
    return ["above_left", "above_right", "left_top", "right_top", "below_left", "below_right"]


def _draw_label_box(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    bg_bgr: tuple[int, int, int],
    cfg: OverlayConfig,
    *,
    font_scale: float,
    font_thickness: int,
) -> None:
    x1, y1, x2, y2 = rect
    bg = _pick_color_for_frame(img, bg_bgr, 255)
    fg = _text_color_for_bgr(img, bg_bgr)

    cv.rectangle(img, (x1, y1), (x2, y2), bg, thickness=-1)

    (tw, th), baseline = cv.getTextSize(text, cfg.font, font_scale, font_thickness)
    text_x = x1 + cfg.label_pad_x
    text_y = y1 + cfg.label_pad_y + th
    cv.putText(img, text, (text_x, text_y), cfg.font, font_scale, fg, font_thickness, cv.LINE_AA)


def _format_short_label(det: Detection, draw_scores: bool) -> str:
    short = _SHORT_LABELS.get(det.label, det.label)
    if draw_scores:
        return f"{short} {det.score:.2f}"
    return short


def _font_for_label(cfg: OverlayConfig, label: str) -> tuple[float, int]:
    if label in {"USB_PORT", "JST_CONNECTOR", "RESET_BUTTON"}:
        return cfg.small_font_scale, cfg.small_font_thickness
    return cfg.font_scale, cfg.font_thickness


def _place_count_labels(
    img: np.ndarray,
    detections: list[Detection],
    cfg: OverlayConfig,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    if not cfg.show_counts:
        return

    counts = count_by_label(detections)
    if not counts:
        return

    frame_h, frame_w = img.shape[:2]
    x = cfg.count_x
    y = cfg.count_y

    order = ["BOARD", "ESP32", "USB_PORT", "JST_CONNECTOR", "RESET_BUTTON"]
    ordered_items = [(label, counts[label]) for label in order if label in counts]
    ordered_items += [(label, count) for label, count in counts.items() if label not in dict(ordered_items)]

    for label, count in ordered_items:
        text = f"{_SHORT_LABELS.get(label, label)}: {count}"
        bg_bgr = _LABEL_COLORS.get(label, (0, 255, 0))

        (tw, th), baseline = cv.getTextSize(
            text,
            cfg.font,
            cfg.count_font_scale,
            cfg.count_font_thickness,
        )
        label_w = tw + 2 * cfg.label_pad_x
        label_h = th + baseline + 2 * cfg.label_pad_y

        rect = _clamp_rect_to_frame((x, y, x + label_w, y + label_h), frame_w, frame_h)

        while any(_boxes_intersect(rect, occ) for occ in occupied):
            y += label_h + cfg.count_row_gap
            rect = _clamp_rect_to_frame((x, y, x + label_w, y + label_h), frame_w, frame_h)

        _draw_label_box(
            img,
            rect,
            text,
            bg_bgr,
            cfg,
            font_scale=cfg.count_font_scale,
            font_thickness=cfg.count_font_thickness,
        )
        occupied.append(rect)
        y += label_h + cfg.count_row_gap


def _place_detection_label(
    img: np.ndarray,
    det: Detection,
    cfg: OverlayConfig,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    frame_h, frame_w = img.shape[:2]
    text = _format_short_label(det, cfg.draw_scores)
    bg_bgr = _LABEL_COLORS.get(det.label, (0, 255, 0))
    font_scale, font_thickness = _font_for_label(cfg, det.label)

    (tw, th), baseline = cv.getTextSize(text, cfg.font, font_scale, font_thickness)
    label_w = tw + 2 * cfg.label_pad_x
    label_h = th + baseline + 2 * cfg.label_pad_y

    bbox = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
    rects = _named_candidate_rects(
        bbox,
        label_w,
        label_h,
        frame_w,
        frame_h,
        cfg.label_gap,
    )

    chosen = None
    for name in _ordered_candidate_names(det):
        rect = rects[name]
        if not any(_boxes_intersect(rect, occ) for occ in occupied):
            chosen = rect
            break

    if chosen is None:
        # fallback: choose candidate with minimum overlap area
        best_rect = None
        best_penalty = None
        for name in _ordered_candidate_names(det):
            rect = rects[name]
            penalty = 0
            rx1, ry1, rx2, ry2 = rect
            for occ in occupied:
                ox1, oy1, ox2, oy2 = occ
                ix1 = max(rx1, ox1)
                iy1 = max(ry1, oy1)
                ix2 = min(rx2, ox2)
                iy2 = min(ry2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    penalty += (ix2 - ix1) * (iy2 - iy1)
            if best_penalty is None or penalty < best_penalty:
                best_penalty = penalty
                best_rect = rect
        chosen = best_rect if best_rect is not None else rects["above_left"]

    _draw_label_box(
        img,
        chosen,
        text,
        bg_bgr,
        cfg,
        font_scale=font_scale,
        font_thickness=font_thickness,
    )
    occupied.append(chosen)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    fps: Optional[float] = None,
    debug: bool = False,
    cfg: OverlayConfig = OverlayConfig(),
) -> np.ndarray:
    vis = frame.copy()
    detections = list(detections)

    if fps is not None:
        fps_color = _pick_color_for_frame(vis, (255, 255, 255), 255)
        cv.putText(vis, f"FPS: {fps:4.1f}", (12, 26), cfg.font, 0.8, fps_color, 2, cv.LINE_AA)

    # draw boxes first
    for det in detections:
        color = _label_color(vis, det.label)
        cv.rectangle(
            vis,
            (det.bbox.x1, det.bbox.y1),
            (det.bbox.x2, det.bbox.y2),
            color,
            thickness=cfg.thickness,
        )

    occupied: list[tuple[int, int, int, int]] = []

    # left count panel first
    _place_count_labels(vis, detections, cfg, occupied)

    # draw labels: small boxes first, BOARD last
    # this helps prevent the BOARD label from stealing useful nearby space
    label_dets = sorted(
        detections,
        key=lambda d: (d.label == "BOARD", d.bbox.area()),
    )

    for det in label_dets:
        _place_detection_label(vis, det, cfg, occupied)

    return vis