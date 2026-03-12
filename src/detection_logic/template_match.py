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
    score_threshold: float = 0.72
    scales: tuple[float, ...] = (0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25)
    nms_iou_threshold: float = 0.25
    max_candidates_per_scale: int = 50
    max_detections: int = 10
    min_template_size: int = 16
    top_k: int | None = 1
    use_edges: bool = True
    edge_weight: float = 0.35
    # Optional ROI in normalized coordinates on the *warped* board.
    # Format: (x, y, w, h), values in [0.0, 1.0].
    roi: tuple[float, float, float, float] | None = None


class TemplateMatcher(Detector):
    """
    Multi-scale template matcher with optional gray/edge score fusion and ROI search.

    Why this version is better for your project:
    - Gray matching uses texture and intensity.
    - Edge matching is more robust to lighting changes.
    - Fusing both helps on real camera photos.
    - Optional ROI makes the future board-aware workflow much faster.
    """

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError("templates_gray is empty.")

        self._cfg = cfg
        self._templates: list[tuple[np.ndarray, np.ndarray]] = []
        for t in templates_gray:
            gray = self._ensure_gray_uint8(t)
            edge = self._compute_edges(gray)
            self._templates.append((gray, edge))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray_full = self._prepare_gray(frame)
        edge_full = self._compute_edges(gray_full)

        gray, edge, x_off, y_off = self._crop_roi(gray_full, edge_full)
        h_frame, w_frame = gray.shape[:2]

        candidates: list[Detection] = []

        for tmpl_gray, tmpl_edge in self._templates:
            for scale in self._cfg.scales:
                t_gray = self._resize_template(tmpl_gray, scale)
                t_edge = self._resize_template(tmpl_edge, scale)
                if t_gray is None or t_edge is None:
                    continue

                th, tw = t_gray.shape[:2]
                if th > h_frame or tw > w_frame:
                    continue

                resp_gray = cv.matchTemplate(gray, t_gray, self._cfg.method).astype(np.float32)
                if self._cfg.use_edges:
                    resp_edge = cv.matchTemplate(edge, t_edge, self._cfg.method).astype(np.float32)
                    resp = (1.0 - self._cfg.edge_weight) * resp_gray + self._cfg.edge_weight * resp_edge
                else:
                    resp = resp_gray

                hits = self._local_maxima(resp, self._cfg.score_threshold)
                if hits is None:
                    continue

                xs, ys, scores = hits
                if scores.size > self._cfg.max_candidates_per_scale:
                    k = self._cfg.max_candidates_per_scale
                    idx = np.argpartition(scores, -k)[-k:]
                    xs, ys, scores = xs[idx], ys[idx], scores[idx]

                for x, y, s in zip(xs, ys, scores):
                    bbox = BBox(
                        int(x + x_off),
                        int(y + y_off),
                        int(x + x_off + tw),
                        int(y + y_off + th),
                    )
                    candidates.append(Detection(self._cfg.label, float(s), bbox))

        detections = nms_detections(
            candidates,
            iou_threshold=self._cfg.nms_iou_threshold,
            max_detections=self._cfg.max_detections,
        )

        if self._cfg.top_k is not None and len(detections) > self._cfg.top_k:
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[: self._cfg.top_k]

        return detections

    def _crop_roi(
        self,
        gray: np.ndarray,
        edge: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        if self._cfg.roi is None:
            return gray, edge, 0, 0

        h, w = gray.shape[:2]
        rx, ry, rw, rh = self._cfg.roi
        x1 = max(0, min(w - 1, int(round(rx * w))))
        y1 = max(0, min(h - 1, int(round(ry * h))))
        x2 = max(x1 + 1, min(w, int(round((rx + rw) * w))))
        y2 = max(y1 + 1, min(h, int(round((ry + rh) * h))))
        return gray[y1:y2, x1:x2], edge[y1:y2, x1:x2], x1, y1

    @staticmethod
    def _local_maxima(resp: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        kernel = np.ones((3, 3), np.uint8)
        resp_dil = cv.dilate(resp, kernel)
        mask = (resp >= threshold) & (resp == resp_dil)

        ys, xs = np.where(mask)
        if xs.size == 0:
            return None

        scores = resp[ys, xs]
        return xs, ys, scores

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = self._ensure_gray_uint8(frame)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        gray = cv.equalizeHist(gray)
        return gray

    @staticmethod
    def _compute_edges(gray: np.ndarray) -> np.ndarray:
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        return cv.Canny(blur, 40, 120)

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
