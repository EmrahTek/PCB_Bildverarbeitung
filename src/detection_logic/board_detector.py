from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2 as cv
import numpy as np

from src.detection_logic.template_match import TemplateMatcher, TemplateMatchConfig
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class BoardDetectorConfig:
    label: str = "BOARD"
    score_threshold: float = 0.66
    scales: tuple[float, ...] = (0.18, 0.22, 0.26, 0.30, 0.36, 0.42, 0.50, 0.60)
    top_k: int = 1
    nms_iou_threshold: float = 0.20
    use_edges: bool = True
    edge_weight: float = 0.35
    allow_tracking: bool = True
    search_margin_px: int = 96
    roi_expand_ratio: float = 0.22
    tracking_max_misses: int = 10


class BoardDetector:
    """Board-first detector.

    It wraps the generic TemplateMatcher but adds a tiny amount of board-specific
    sanity filtering so that random monitor / window / hand regions are less
    likely to be accepted as the FireBeetle board.
    """

    def __init__(self, templates: Iterable[np.ndarray], cfg: BoardDetectorConfig) -> None:
        self._cfg = cfg
        self._matcher = TemplateMatcher(
            templates,
            TemplateMatchConfig(
                label=cfg.label,
                score_threshold=cfg.score_threshold,
                scales=cfg.scales,
                nms_iou_threshold=cfg.nms_iou_threshold,
                top_k=cfg.top_k,
                use_edges=cfg.use_edges,
                edge_weight=cfg.edge_weight,
                allow_tracking=cfg.allow_tracking,
                search_margin_px=cfg.search_margin_px,
                roi_expand_ratio=cfg.roi_expand_ratio,
                tracking_max_misses=cfg.tracking_max_misses,
            ),
        )

    def reset_tracking(self) -> None:
        self._matcher.reset_tracking()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raw = self._matcher.detect(frame)
        filtered = [d for d in raw if self._is_plausible_board(d.bbox, frame.shape[1], frame.shape[0])]
        return filtered[:1]

    @staticmethod
    def crop(frame: np.ndarray, board_bbox: BBox, margin_ratio: float = 0.03) -> np.ndarray:
        pad_x = int(board_bbox.w * margin_ratio)
        pad_y = int(board_bbox.h * margin_ratio)
        x1 = max(0, board_bbox.x - pad_x)
        y1 = max(0, board_bbox.y - pad_y)
        x2 = min(frame.shape[1], board_bbox.x + board_bbox.w + pad_x)
        y2 = min(frame.shape[0], board_bbox.y + board_bbox.h + pad_y)
        return frame[y1:y2, x1:x2].copy()

    def _is_plausible_board(self, bbox: BBox, frame_w: int, frame_h: int) -> bool:
        area_ratio = (bbox.w * bbox.h) / float(frame_w * frame_h)
        if area_ratio < 0.01 or area_ratio > 0.80:
            return False

        ratio = max(bbox.w, bbox.h) / max(1.0, min(bbox.w, bbox.h))
        if ratio < 1.5 or ratio > 5.5:
            return False

        # Reject clearly edge-clipped huge boxes that often come from window/monitor borders.
        touches_many_edges = int(bbox.x <= 2) + int(bbox.y <= 2) + int(bbox.x + bbox.w >= frame_w - 2) + int(bbox.y + bbox.h >= frame_h - 2)
        if area_ratio > 0.45 and touches_many_edges >= 2:
            return False

        return True
