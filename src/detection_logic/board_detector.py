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


@dataclass(frozen=True)
class Esp32InBoardConfig:
    label: str = "ESP32"
    score_threshold: float = 0.70
    scales: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30)
    top_k: int = 1
    nms_iou_threshold: float = 0.20
    use_edges: bool = True
    edge_weight: float = 0.25
    roi_expand_ratio: float = 0.10
    search_margin_px: int = 48
    allow_tracking: bool = False
    tracking_max_misses: int = 6
    board_margin_ratio: float = 0.03
    search_rel_x: float = 0.36
    search_rel_y: float = 0.42
    search_rel_w: float = 0.56
    search_rel_h: float = 0.54


class BoardDetector:
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
    def crop(frame: np.ndarray, board_bbox: BBox, margin_ratio: float = 0.03) -> tuple[np.ndarray, tuple[int, int]]:
        pad_x = int(board_bbox.w * margin_ratio)
        pad_y = int(board_bbox.h * margin_ratio)
        x1 = max(0, board_bbox.x - pad_x)
        y1 = max(0, board_bbox.y - pad_y)
        x2 = min(frame.shape[1], board_bbox.x + board_bbox.w + pad_x)
        y2 = min(frame.shape[0], board_bbox.y + board_bbox.h + pad_y)
        return frame[y1:y2, x1:x2].copy(), (x1, y1)

    def _is_plausible_board(self, bbox: BBox, frame_w: int, frame_h: int) -> bool:
        area_ratio = (bbox.w * bbox.h) / float(frame_w * frame_h)
        if area_ratio < 0.01 or area_ratio > 0.85:
            return False

        ratio = max(bbox.w, bbox.h) / max(1.0, min(bbox.w, bbox.h))
        if ratio < 1.35 or ratio > 6.0:
            return False

        touches_many_edges = int(bbox.x <= 2) + int(bbox.y <= 2) + int(bbox.x + bbox.w >= frame_w - 2) + int(bbox.y + bbox.h >= frame_h - 2)
        if area_ratio > 0.55 and touches_many_edges >= 2:
            return False
        return True


class BoardEsp32Detector:
    """First detect BOARD on the full frame, then detect ESP32 only inside the board ROI."""

    def __init__(
        self,
        board_templates: Iterable[np.ndarray],
        board_cfg: BoardDetectorConfig,
        esp32_templates: Iterable[np.ndarray],
        esp32_cfg: Esp32InBoardConfig,
    ) -> None:
        self._board = BoardDetector(board_templates, board_cfg)
        self._esp32_cfg = esp32_cfg
        self._esp32 = TemplateMatcher(
            esp32_templates,
            TemplateMatchConfig(
                label=esp32_cfg.label,
                score_threshold=esp32_cfg.score_threshold,
                scales=esp32_cfg.scales,
                nms_iou_threshold=esp32_cfg.nms_iou_threshold,
                top_k=esp32_cfg.top_k,
                use_edges=esp32_cfg.use_edges,
                edge_weight=esp32_cfg.edge_weight,
                allow_tracking=esp32_cfg.allow_tracking,
                search_margin_px=esp32_cfg.search_margin_px,
                roi_expand_ratio=esp32_cfg.roi_expand_ratio,
                tracking_max_misses=esp32_cfg.tracking_max_misses,
            ),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        board_dets = self._board.detect(frame)
        if not board_dets:
            self._esp32.reset_tracking()
            return []

        board_det = board_dets[0]
        board_crop, (ox, oy) = self._board.crop(frame, board_det.bbox, margin_ratio=self._esp32_cfg.board_margin_ratio)

        roi = self._esp32_search_roi(board_crop)
        board_roi = board_crop[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]
        esp_dets_local = self._esp32.detect(board_roi)

        translated: list[Detection] = [board_det]
        for det in esp_dets_local:
            bb = det.bbox
            translated_bbox = BBox(ox + roi.x + bb.x, oy + roi.y + bb.y, bb.w, bb.h)
            if self._bbox_inside_board(translated_bbox, board_det.bbox):
                translated.append(Detection(det.label, det.score, translated_bbox))

        return translated

    def _esp32_search_roi(self, board_crop: np.ndarray) -> BBox:
        h, w = board_crop.shape[:2]
        x = int(self._esp32_cfg.search_rel_x * w)
        y = int(self._esp32_cfg.search_rel_y * h)
        rw = int(self._esp32_cfg.search_rel_w * w)
        rh = int(self._esp32_cfg.search_rel_h * h)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        return BBox(x, y, rw, rh)

    @staticmethod
    def _bbox_inside_board(candidate: BBox, board: BBox) -> bool:
        cx1, cy1 = candidate.x, candidate.y
        cx2, cy2 = candidate.x + candidate.w, candidate.y + candidate.h
        bx1, by1 = board.x, board.y
        bx2, by2 = board.x + board.w, board.y + board.h
        return cx1 >= bx1 and cy1 >= by1 and cx2 <= bx2 and cy2 <= by2
