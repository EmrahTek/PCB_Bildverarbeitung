from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import nms_detections, translate_detections
from src.preprocessing.filters import preprocess_for_template
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    label: str = "ESP32"
    method: int = cv.TM_CCOEFF_NORMED
    score_threshold: float = 0.80
    scales: tuple[float, ...] = (0.18, 0.20, 0.22, 0.25, 0.28, 0.30)
    nms_iou_threshold: float = 0.20
    max_candidates_per_scale: int = 40
    max_detections: int = 5
    min_template_size: int = 12
    top_k: int | None = 1
    use_clahe: bool = True
    edge_mode: bool = True
    blur_ksize: int = 3
    search_roi: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class PreparedTemplate:
    scale: float
    image: np.ndarray
    width: int
    height: int


class TemplateMatcher(Detector):
    """Fast template matcher with precomputed scaled templates and optional ROI search."""

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError("templates_gray is empty")
        self._cfg = cfg
        self._prepared_templates = self._prepare_templates(templates_gray)
        if not self._prepared_templates:
            raise ValueError("No usable prepared templates were generated")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        search_img, dx, dy = self._crop_to_roi(frame)
        proc = preprocess_for_template(
            search_img,
            use_clahe=self._cfg.use_clahe,
            edge_mode=self._cfg.edge_mode,
            blur_ksize=self._cfg.blur_ksize,
        )
        h_frame, w_frame = proc.shape[:2]

        candidates: list[Detection] = []
        for tmpl in self._prepared_templates:
            if tmpl.height > h_frame or tmpl.width > w_frame:
                continue

            response = cv.matchTemplate(proc, tmpl.image, self._cfg.method).astype(np.float32)
            response_dilated = cv.dilate(response, np.ones((3, 3), np.uint8))
            mask = (response >= self._cfg.score_threshold) & (response == response_dilated)
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue

            scores = response[ys, xs]
            if scores.size > self._cfg.max_candidates_per_scale:
                k = self._cfg.max_candidates_per_scale
                idx = np.argpartition(scores, -k)[-k:]
                xs, ys, scores = xs[idx], ys[idx], scores[idx]

            for x, y, score in zip(xs, ys, scores):
                candidates.append(
                    Detection(
                        label=self._cfg.label,
                        score=float(score),
                        bbox=BBox(int(x), int(y), int(x + tmpl.width), int(y + tmpl.height)),
                    )
                )

        detections = nms_detections(
            candidates,
            iou_threshold=self._cfg.nms_iou_threshold,
            max_detections=self._cfg.max_detections,
        )

        if self._cfg.top_k is not None:
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[: self._cfg.top_k]

        return translate_detections(detections, dx, dy)

    def _prepare_templates(self, templates_gray: list[np.ndarray]) -> list[PreparedTemplate]:
        prepared: list[PreparedTemplate] = []
        for template in templates_gray:
            template_u8 = self._ensure_gray_uint8(template)
            for scale in self._cfg.scales:
                if scale <= 0:
                    continue
                h, w = template_u8.shape[:2]
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                if new_w < self._cfg.min_template_size or new_h < self._cfg.min_template_size:
                    continue
                resized = cv.resize(template_u8, (new_w, new_h), interpolation=cv.INTER_AREA)
                prepared_img = preprocess_for_template(
                    resized,
                    use_clahe=self._cfg.use_clahe,
                    edge_mode=self._cfg.edge_mode,
                    blur_ksize=self._cfg.blur_ksize,
                )
                prepared.append(PreparedTemplate(scale=scale, image=prepared_img, width=new_w, height=new_h))
        return prepared

    def _crop_to_roi(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self._cfg.search_roi is None:
            return frame, 0, 0
        x, y, w, h = self._cfg.search_roi
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        x2 = max(x + 1, min(x + w, w_frame))
        y2 = max(y + 1, min(y + h, h_frame))
        return frame[y:y2, x:x2], x, y

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
