from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import nms_detections
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    label: str = "ESP32"
    method: int = cv.TM_CCOEFF_NORMED
    score_threshold: float = 0.66
    scales: tuple[float, ...] = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)
    nms_iou_threshold: float = 0.25
    max_candidates_per_template: int = 8
    max_detections: int = 8
    min_template_size: int = 14
    top_k: int | None = 1
    use_clahe: bool = True
    blur_ksize: int = 3
    local_max_kernel: int = 5


class TemplateMatcher(Detector):
    """Fast multi-scale template matcher with precomputed scaled templates."""

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError("templates_gray is empty.")
        self._cfg = cfg
        self._scaled_templates: list[np.ndarray] = []
        for tmpl in templates_gray:
            base = self._prepare_template(tmpl)
            for scale in cfg.scales:
                resized = self._resize_template(base, scale)
                if resized is not None:
                    self._scaled_templates.append(resized)
        if not self._scaled_templates:
            raise ValueError("No usable scaled templates could be built.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = self._prepare_frame(frame)
        h_frame, w_frame = gray.shape[:2]
        candidates: list[Detection] = []
        kernel_size = max(3, int(self._cfg.local_max_kernel) | 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        for tmpl in self._scaled_templates:
            th, tw = tmpl.shape[:2]
            if th > h_frame or tw > w_frame:
                continue

            resp = cv.matchTemplate(gray, tmpl, self._cfg.method)
            resp_f = resp.astype(np.float32)
            resp_dil = cv.dilate(resp_f, kernel)
            mask = (resp_f >= self._cfg.score_threshold) & (resp_f == resp_dil)
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue

            scores = resp_f[ys, xs]
            if scores.size > self._cfg.max_candidates_per_template:
                k = self._cfg.max_candidates_per_template
                idx = np.argpartition(scores, -k)[-k:]
                xs, ys, scores = xs[idx], ys[idx], scores[idx]

            for x, y, s in zip(xs, ys, scores):
                bbox = BBox(int(x), int(y), int(x + tw), int(y + th))
                candidates.append(Detection(self._cfg.label, float(s), bbox))

        detections = nms_detections(
            candidates,
            iou_threshold=self._cfg.nms_iou_threshold,
            max_detections=self._cfg.max_detections,
        )
        if self._cfg.top_k is not None and len(detections) > self._cfg.top_k:
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[: self._cfg.top_k]
        return detections

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = self._ensure_gray_uint8(frame)
        if self._cfg.use_clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        k = max(1, int(self._cfg.blur_ksize) | 1)
        if k > 1:
            gray = cv.GaussianBlur(gray, (k, k), 0)
        return gray

    def _prepare_template(self, tmpl: np.ndarray) -> np.ndarray:
        return self._prepare_frame(tmpl)

    @staticmethod
    def _ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
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
        if scale <= 0:
            return None
        h, w = tmpl.shape[:2]
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w < self._cfg.min_template_size or new_h < self._cfg.min_template_size:
            return None
        return cv.resize(tmpl, (new_w, new_h), interpolation=cv.INTER_AREA)
