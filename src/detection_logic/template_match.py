from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    label: str
    score_threshold: float = 0.72
    scales: tuple[float, ...] = (1.0,)
    nms_iou_threshold: float = 0.20
    top_k: int = 1
    use_edges: bool = False
    edge_weight: float = 0.30
    roi_expand_ratio: float = 0.20
    search_margin_px: int = 72
    allow_tracking: bool = True
    tracking_max_misses: int = 8


@dataclass(frozen=True)
class _PreparedTemplate:
    gray: np.ndarray
    edges: np.ndarray | None


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    return cv.Canny(gray, 70, 180)


def _make_bbox(x: int, y: int, w: int, h: int) -> BBox:
    return BBox(int(x), int(y), int(w), int(h))


def _clip_bbox(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> BBox:
    x = max(0, min(int(x), frame_w - 1))
    y = max(0, min(int(y), frame_h - 1))
    w = max(1, min(int(w), frame_w - x))
    h = max(1, min(int(h), frame_h - y))
    return _make_bbox(x, y, w, h)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    union = a.w * a.h + b.w * b.h - inter_area
    return inter_area / union if union > 0 else 0.0


def _nms(detections: list[Detection], iou_threshold: float, top_k: int) -> list[Detection]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []

    for det in detections:
        if all(_bbox_iou(det.bbox, prev.bbox) < iou_threshold for prev in kept):
            kept.append(det)
            if len(kept) >= top_k:
                break
    return kept


class TemplateMatcher:
    """Generic multi-scale template matcher with optional local ROI tracking."""

    def __init__(self, templates: Iterable[np.ndarray], cfg: TemplateMatchConfig) -> None:
        raw_templates = [t for t in templates if t is not None and t.size > 0]
        if not raw_templates:
            raise ValueError(f"No valid templates for label={cfg.label}")

        self._cfg = cfg
        self._templates: list[_PreparedTemplate] = []
        for img in raw_templates:
            gray = _to_gray(img)
            edges = _edge_map(gray) if cfg.use_edges else None
            self._templates.append(_PreparedTemplate(gray=gray, edges=edges))

        self._last_bbox: BBox | None = None
        self._misses = 0

    def reset_tracking(self) -> None:
        self._last_bbox = None
        self._misses = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        frame_gray = _to_gray(frame)
        frame_edges = _edge_map(frame_gray) if self._cfg.use_edges else None
        frame_h, frame_w = frame_gray.shape[:2]

        search_regions = self._make_search_regions(frame_w, frame_h)
        candidates: list[Detection] = []

        for region_idx, roi in enumerate(search_regions):
            roi_gray = frame_gray[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]
            roi_edges = None if frame_edges is None else frame_edges[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]

            region_candidates = self._search_region(roi_gray, roi_edges, roi)
            if region_candidates:
                candidates.extend(region_candidates)
                if region_idx == 0 and self._last_bbox is not None:
                    break

        detections = _nms(candidates, self._cfg.nms_iou_threshold, self._cfg.top_k)

        if detections:
            self._last_bbox = detections[0].bbox
            self._misses = 0
        else:
            self._misses += 1
            if self._misses >= self._cfg.tracking_max_misses:
                self._last_bbox = None

        return detections

    def _make_search_regions(self, frame_w: int, frame_h: int) -> list[BBox]:
        full = _make_bbox(0, 0, frame_w, frame_h)
        if not self._cfg.allow_tracking or self._last_bbox is None:
            return [full]

        pad_x = max(self._cfg.search_margin_px, int(self._last_bbox.w * self._cfg.roi_expand_ratio))
        pad_y = max(self._cfg.search_margin_px, int(self._last_bbox.h * self._cfg.roi_expand_ratio))
        local = _clip_bbox(
            x=self._last_bbox.x - pad_x,
            y=self._last_bbox.y - pad_y,
            w=self._last_bbox.w + 2 * pad_x,
            h=self._last_bbox.h + 2 * pad_y,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        return [local, full]

    def _search_region(self, roi_gray: np.ndarray, roi_edges: np.ndarray | None, roi_bbox: BBox) -> list[Detection]:
        roi_h, roi_w = roi_gray.shape[:2]
        out: list[Detection] = []

        for tmpl in self._templates:
            th0, tw0 = tmpl.gray.shape[:2]
            for scale in self._cfg.scales:
                tw = max(8, int(round(tw0 * scale)))
                th = max(8, int(round(th0 * scale)))
                if tw >= roi_w or th >= roi_h:
                    continue

                interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
                templ_gray = cv.resize(tmpl.gray, (tw, th), interpolation=interp)
                gray_map = cv.matchTemplate(roi_gray, templ_gray, cv.TM_CCOEFF_NORMED)
                _minv, maxv_gray, _minl, maxl_gray = cv.minMaxLoc(gray_map)
                score = float(maxv_gray)
                x0, y0 = maxl_gray

                if self._cfg.use_edges and roi_edges is not None and tmpl.edges is not None:
                    templ_edges = cv.resize(tmpl.edges, (tw, th), interpolation=interp)
                    edge_map = cv.matchTemplate(roi_edges, templ_edges, cv.TM_CCOEFF_NORMED)
                    _emin, maxv_edge, _eminl, maxl_edge = cv.minMaxLoc(edge_map)
                    edge_weight = float(np.clip(self._cfg.edge_weight, 0.0, 1.0))
                    score = (1.0 - edge_weight) * float(maxv_gray) + edge_weight * float(maxv_edge)
                    if maxv_edge > maxv_gray:
                        x0, y0 = maxl_edge

                if score < self._cfg.score_threshold:
                    continue

                bbox = _make_bbox(roi_bbox.x + int(x0), roi_bbox.y + int(y0), tw, th)
                out.append(Detection(self._cfg.label, float(score), bbox))

        return out
