from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.template_match import TemplateMatcher
from src.preprocessing.geometry import BoardWarpConfig, detect_and_warp_board
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class BoardFirstEsp32Config:
    board_cfg: BoardWarpConfig = BoardWarpConfig()
    board_label: str = "BOARD"
    esp32_label: str = "ESP32"
    fallback_direct_esp32: bool = False
    esp32_min_score_after_warp: float = 0.62


class BoardFirstEsp32Detector(Detector):
    """
    Robust reference detector:
    1) localise board by geometry/homography
    2) warp board to canonical view
    3) detect ESP32 only inside warped board
    4) map ESP32 bbox back to original image coordinates
    """

    def __init__(
        self,
        esp32_in_board_matcher: TemplateMatcher,
        cfg: BoardFirstEsp32Config = BoardFirstEsp32Config(),
        *,
        direct_fallback_matcher: TemplateMatcher | None = None,
    ) -> None:
        self._esp32_in_board = esp32_in_board_matcher
        self._fallback = direct_fallback_matcher
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = detect_and_warp_board(frame, self._cfg.board_cfg)
        if result is None:
            if self._cfg.fallback_direct_esp32 and self._fallback is not None:
                return self._fallback.detect(frame)
            return []

        detections: list[Detection] = []
        board_bbox = self._quad_to_bbox(result.quad, frame.shape)
        detections.append(Detection(self._cfg.board_label, max(0.60, min(1.0, result.score + 0.15)), board_bbox))

        warped_dets = self._esp32_in_board.detect(result.warped)
        if not warped_dets:
            return detections

        H_inv = np.linalg.inv(result.homography)
        for det in warped_dets:
            if det.score < self._cfg.esp32_min_score_after_warp:
                continue
            mapped = self._map_bbox_back(det.bbox, H_inv, frame.shape)
            if mapped is None or mapped.area() <= 0:
                continue
            detections.append(Detection(self._cfg.esp32_label, det.score, mapped))

        return detections

    @staticmethod
    def _quad_to_bbox(quad: np.ndarray, frame_shape: tuple[int, ...]) -> BBox:
        h, w = frame_shape[:2]
        x1 = int(max(0, np.floor(np.min(quad[:, 0]))))
        y1 = int(max(0, np.floor(np.min(quad[:, 1]))))
        x2 = int(min(w - 1, np.ceil(np.max(quad[:, 0]))))
        y2 = int(min(h - 1, np.ceil(np.max(quad[:, 1]))))
        return BBox(x1, y1, x2, y2)

    @staticmethod
    def _map_bbox_back(bbox: BBox, H_inv: np.ndarray, frame_shape: tuple[int, ...]) -> BBox | None:
        h, w = frame_shape[:2]
        corners = np.array(
            [[bbox.x1, bbox.y1], [bbox.x2, bbox.y1], [bbox.x2, bbox.y2], [bbox.x1, bbox.y2]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        proj = cv.perspectiveTransform(corners, H_inv).reshape(-1, 2)
        x1 = int(max(0, np.floor(np.min(proj[:, 0]))))
        y1 = int(max(0, np.floor(np.min(proj[:, 1]))))
        x2 = int(min(w - 1, np.ceil(np.max(proj[:, 0]))))
        y2 = int(min(h - 1, np.ceil(np.max(proj[:, 1]))))
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox(x1, y1, x2, y2)
