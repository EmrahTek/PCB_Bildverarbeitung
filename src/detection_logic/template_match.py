# Baseline: matchTemplate + Multi-Scale
# Baseline-Detektion per Template Matching (Multi-Scale), optional auf Gray oder Edges.

"""
template_match.py

This module implements template matching based detection using OpenCV's matchTemplate.
It supports:
- matching multiple templates per label
- multi-scale matching via image pyramids
- candidate extraction from score maps

Inputs:
- Preprocessed image (grayscale or edge map)
- Template images (grayscale or edge map)
- Matching configuration (method, thresholds, scales)

Outputs:
- list[Detection] with label, score, and bounding boxes

Zu implementierende Funktionen

    build_pyramid(image, scales) -> list[tuple[scale, image_scaled]]

    match_single_template(image, template, method) -> score_map

    extract_candidates(score_map, threshold) -> list[BBox+score]

    multi_scale_template_match(image, templates, scales, threshold) -> list[Detection]

    (Optional) prepare_image_for_matching(image, mode="gray|edges") -> img

matchTemplate tutorial:
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

matchTemplate reference:
https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html

Image pyramids (search terms):
"opencv image pyramid python"

"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import nms_detections
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    """
    Config for multi-scale template matching.

    Tuning knobs:
      - score_threshold: main control for false positives/negatives
      - scales: handle distance/zoom variation
      - nms_iou_threshold: how aggressively to merge overlaps
      - top_k: keep only the best K detections after NMS (ESP32 usually 1)
    """
    label: str = "ESP32"
    method: int = cv.TM_CCOEFF_NORMED
    score_threshold: float = 0.70
    scales: tuple[float, ...] = (
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
    )
    nms_iou_threshold: float = 0.3
    max_candidates_per_scale: int = 200
    max_detections: int = 20
    min_template_size: int = 12  # px
    top_k: int | None = 1


class TemplateMatcher(Detector):
    """
    Detector using OpenCV matchTemplate across multiple scales.

    Works best if templates are tight crops of the component.
    """

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError("templates_gray is empty.")
        self._templates = [self._ensure_gray_uint8(t) for t in templates_gray]
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = self._prepare_frame(frame)
        h_frame, w_frame = gray.shape[:2]

        candidates: list[Detection] = []

        for tmp in self._templates:
            for scale in self._cfg.scales:
                resized = self._resize_template(tmp, scale)
                if resized is None:
                    continue

                th, tw = resized.shape[:2]

                # IMPORTANT: allow template == frame size (use '>' not '>=')
                if th > h_frame or tw > w_frame:
                    continue

                # Response map
                resp = cv.matchTemplate(gray, resized, self._cfg.method)

                # Keep only local maxima above threshold (reduces duplicates)
                resp_f = resp.astype(np.float32)
                kernel = np.ones((3, 3), np.uint8)
                resp_dil = cv.dilate(resp_f, kernel)
                mask = (resp_f >= self._cfg.score_threshold) & (resp_f == resp_dil)

                ys, xs = np.where(mask)
                if xs.size == 0:
                    continue

                scores = resp_f[ys, xs]

                # Speed guard: keep only top-K hits per scale
                if scores.size > self._cfg.max_candidates_per_scale:
                    k = self._cfg.max_candidates_per_scale
                    idx = np.argpartition(scores, -k)[-k:]
                    xs, ys, scores = xs[idx], ys[idx], scores[idx]

                for x, y, s in zip(xs, ys, scores):
                    bbox = BBox(int(x), int(y), int(x + tw), int(y + th))
                    candidates.append(Detection(self._cfg.label, float(s), bbox))

        # Merge overlaps (NMS)
        detections = nms_detections(
            candidates,
            iou_threshold=self._cfg.nms_iou_threshold,
            max_detections=self._cfg.max_detections,
        )

        # Keep only the best K detections (ESP32 usually exists once on a board)
        if self._cfg.top_k is not None and len(detections) > self._cfg.top_k:
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[: self._cfg.top_k]

        return detections

    # ----- helpers -----

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert to gray uint8 and apply a small blur."""
        gray = self._ensure_gray_uint8(frame)
        return cv.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
        """
        Ensure input is grayscale uint8.
        Accepts gray (H,W) or BGR (H,W,3).
        """
        if img.ndim == 2:
            out = img
        elif img.ndim == 3 and img.shape[2] == 3:
            out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if out.dtype != np.uint8:
            out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

        return out

    def _resize_template(self, tmpl: np.ndarray, scale: float) -> np.ndarray | None:
        """Resize template; return None if too small."""
        if scale <= 0:
            return None

        h, w = tmpl.shape[:2]
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        if new_w < self._cfg.min_template_size or new_h < self._cfg.min_template_size:
            return None

        return cv.resize(tmpl, (new_w, new_h), interpolation=cv.INTER_AREA)
    
    