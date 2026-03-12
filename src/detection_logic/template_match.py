from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import nms_detections
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    """
    Configuration for a single component matcher.

    Notes:
    - max_candidates_per_scale is intentionally kept small for speed.
    - use_tracking speeds up webcam/video by searching around the previous result first.
    - use_edges can help smaller components like USB/JST/RESET, but costs extra time.
    """

    label: str = "ESP32"
    method: int = cv.TM_CCOEFF_NORMED
    score_threshold: float = 0.72
    scales: tuple[float, ...] = (0.60, 0.75, 0.90, 1.05, 1.20)
    nms_iou_threshold: float = 0.25
    max_candidates_per_scale: int = 1
    max_detections: int = 5
    min_template_size: int = 12
    top_k: int | None = 1
    use_edges: bool = False
    edge_weight: float = 0.25
    use_tracking: bool = False
    tracking_expansion: float = 1.8
    force_full_frame_every_n: int = 12


class ComponentTemplateMatcher(Detector):
    """
    Fast template matcher for one label.

    Strategy:
    1. Match on grayscale.
    2. Optionally refine score using edges at the same location.
    3. In live mode, first search inside an expanded ROI around the previous hit.
    4. If ROI search fails, fall back to a full-frame search.
    """

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError(f"No templates provided for label={cfg.label}")

        self._cfg = cfg
        self._templates_gray: list[np.ndarray] = []
        self._templates_edge: list[np.ndarray] = []
        for tmp in templates_gray:
            gray = self._ensure_gray_uint8(tmp)
            self._templates_gray.append(gray)
            self._templates_edge.append(self._compute_edges(gray))

        self._last_bbox: BBox | None = None
        self._frames_since_full = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = self._prepare_frame(frame)
        full_h, full_w = gray.shape[:2]
        if full_h == 0 or full_w == 0:
            return []

        edges = self._compute_edges(gray) if self._cfg.use_edges else None

        roi = self._make_tracking_roi(full_w, full_h)
        candidates: list[Detection] = []

        # First try a local ROI when tracking is enabled.
        if roi is not None:
            x1, y1, x2, y2 = roi
            gray_roi = gray[y1:y2, x1:x2]
            edge_roi = edges[y1:y2, x1:x2] if edges is not None else None
            candidates = self._detect_in_gray(gray_roi, edge_roi, offset=(x1, y1))

        # Fallback to full-frame search if ROI failed or tracking is off.
        if not candidates:
            candidates = self._detect_in_gray(gray, edges, offset=(0, 0))
            self._frames_since_full = 0
        else:
            self._frames_since_full += 1

        if not candidates:
            self._last_bbox = None
            return []

        detections = nms_detections(
            candidates,
            iou_threshold=self._cfg.nms_iou_threshold,
            max_detections=self._cfg.max_detections,
        )
        if self._cfg.top_k is not None:
            detections = detections[: self._cfg.top_k]

        self._last_bbox = detections[0].bbox if detections else None
        return detections

    def _detect_in_gray(
        self,
        gray: np.ndarray,
        edges: np.ndarray | None,
        *,
        offset: tuple[int, int],
    ) -> list[Detection]:
        h_frame, w_frame = gray.shape[:2]
        out: list[Detection] = []
        off_x, off_y = offset

        for template_gray, template_edge in zip(self._templates_gray, self._templates_edge):
            for scale in self._cfg.scales:
                scaled_gray = self._resize_template(template_gray, scale)
                if scaled_gray is None:
                    continue

                th, tw = scaled_gray.shape[:2]
                if th > h_frame or tw > w_frame:
                    continue

                response = cv.matchTemplate(gray, scaled_gray, self._cfg.method)
                if response.size == 0:
                    continue

                # Keep only the strongest location per scale for speed.
                score, x, y = self._best_response(response)
                if score < (self._cfg.score_threshold - (0.05 if self._cfg.use_edges else 0.0)):
                    continue

                final_score = score
                if self._cfg.use_edges and edges is not None:
                    scaled_edge = self._resize_template(template_edge, scale)
                    if scaled_edge is not None and scaled_edge.shape == scaled_gray.shape:
                        edge_patch = edges[y:y + th, x:x + tw]
                        if edge_patch.shape == scaled_edge.shape:
                            edge_resp = cv.matchTemplate(edge_patch, scaled_edge, self._cfg.method)
                            if edge_resp.size > 0:
                                edge_score = float(edge_resp[0, 0])
                                final_score = (
                                    (1.0 - self._cfg.edge_weight) * score
                                    + self._cfg.edge_weight * edge_score
                                )

                if final_score < self._cfg.score_threshold:
                    continue

                bbox = BBox(
                    x1=off_x + int(x),
                    y1=off_y + int(y),
                    x2=off_x + int(x + tw),
                    y2=off_y + int(y + th),
                )
                out.append(Detection(label=self._cfg.label, score=float(final_score), bbox=bbox))

        out.sort(key=lambda d: d.score, reverse=True)
        return out[: max(1, self._cfg.max_candidates_per_scale * max(1, len(self._cfg.scales)))]

    def _make_tracking_roi(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
        if not self._cfg.use_tracking or self._last_bbox is None:
            return None
        if self._cfg.force_full_frame_every_n > 0 and self._frames_since_full >= self._cfg.force_full_frame_every_n:
            return None

        box = self._last_bbox
        bw = max(1, box.x2 - box.x1)
        bh = max(1, box.y2 - box.y1)
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0

        expand = max(1.2, float(self._cfg.tracking_expansion))
        half_w = int(round(0.5 * bw * expand))
        half_h = int(round(0.5 * bh * expand))

        x1 = max(0, int(round(cx - half_w)))
        y1 = max(0, int(round(cy - half_h)))
        x2 = min(frame_w, int(round(cx + half_w)))
        y2 = min(frame_h, int(round(cy + half_h)))

        if x2 - x1 < self._cfg.min_template_size or y2 - y1 < self._cfg.min_template_size:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _best_response(response: np.ndarray) -> tuple[float, int, int]:
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
        # For normalized correlation methods, larger is better.
        return float(max_val), int(max_loc[0]), int(max_loc[1])

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        gray = ComponentTemplateMatcher._ensure_gray_uint8(frame)
        # Light normalization makes webcam lighting slightly more stable.
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return gray

    @staticmethod
    def _ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if gray.dtype != np.uint8:
            gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def _resize_template(template: np.ndarray, scale: float) -> np.ndarray | None:
        if scale <= 0:
            return None
        h, w = template.shape[:2]
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w < 12 or new_h < 12:
            return None
        interpolation = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        return cv.resize(template, (new_w, new_h), interpolation=interpolation)

    @staticmethod
    def _compute_edges(gray: np.ndarray) -> np.ndarray:
        return cv.Canny(gray, 60, 160)


class CompositeTemplateMatcher(Detector):
    """Combine several component matchers into one detector."""

    def __init__(self, matchers: list[ComponentTemplateMatcher]) -> None:
        if not matchers:
            raise ValueError("CompositeTemplateMatcher requires at least one matcher")
        self._matchers = matchers

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for matcher in self._matchers:
            detections.extend(matcher.detect(frame))
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections


class TemplateMatcher(ComponentTemplateMatcher):
    """
    Backward-compatible alias.

    Existing project code can still instantiate TemplateMatcher(templates, cfg).
    """

    pass
