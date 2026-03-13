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
    fallback_direct_esp32: bool = True
    esp32_min_score_after_warp: float = 0.60
    direct_board_min_score: float = 0.56
    direct_esp32_min_score: float = 0.68


class BoardFirstEsp32Detector(Detector):
    """
    Three-stage cascade detector:

    1) geometry -> warp board -> detect ESP32 inside warped board
    2) full-frame board template fallback -> crop board -> detect ESP32 in resized crop
    3) full-frame ESP32 fallback -> estimate board from ESP32 anchor

    This is more robust than a pure board-first geometry pipeline.
    """

    def __init__(
        self,
        esp32_in_board_matcher: TemplateMatcher,
        cfg: BoardFirstEsp32Config = BoardFirstEsp32Config(),
        *,
        direct_fallback_matcher: TemplateMatcher | None = None,
        direct_board_matcher: TemplateMatcher | None = None,
    ) -> None:
        self._esp32_in_board = esp32_in_board_matcher
        self._fallback = direct_fallback_matcher
        self._board_full_frame = direct_board_matcher
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
        geom_dets = self._detect_via_geometry(frame)
        if self._has_label(geom_dets, self._cfg.esp32_label):
            return geom_dets

        board_tpl_dets = self._detect_via_board_template(frame)
        if self._has_label(board_tpl_dets, self._cfg.esp32_label):
            return board_tpl_dets

        direct_esp32_dets = self._detect_via_direct_esp32(frame)

        merged = self._merge_board_and_esp32(geom_dets, direct_esp32_dets)
        if merged:
            return merged

        merged = self._merge_board_and_esp32(board_tpl_dets, direct_esp32_dets)
        if merged:
            return merged

        if board_tpl_dets:
            return board_tpl_dets
        if geom_dets:
            return geom_dets
        if direct_esp32_dets:
            return direct_esp32_dets

        return []

    def _detect_via_geometry(self, frame: np.ndarray) -> list[Detection]:
        result = detect_and_warp_board(frame, self._cfg.board_cfg)
        if result is None:
            return []

        detections: list[Detection] = []
        board_bbox = self._quad_to_bbox(result.quad, frame.shape)
        board_score = max(0.60, min(1.0, result.score + 0.15))
        detections.append(Detection(self._cfg.board_label, board_score, board_bbox))

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

    def _detect_via_board_template(self, frame: np.ndarray) -> list[Detection]:
        if self._board_full_frame is None:
            return []

        board_dets = self._board_full_frame.detect(frame)
        if not board_dets:
            return []

        best_board = max(board_dets, key=lambda d: d.score)
        if best_board.score < self._cfg.direct_board_min_score:
            return []

        board_box = self._clip_bbox(best_board.bbox, frame.shape)
        detections: list[Detection] = [
            Detection(self._cfg.board_label, best_board.score, board_box)
        ]

        crop = frame[board_box.y1:board_box.y2, board_box.x1:board_box.x2]
        crop_esp32 = self._detect_esp32_in_axis_aligned_crop(crop, board_box, frame.shape)
        detections.extend(crop_esp32)
        return detections

    def _detect_via_direct_esp32(self, frame: np.ndarray) -> list[Detection]:
        if not self._cfg.fallback_direct_esp32 or self._fallback is None:
            return []

        direct_dets = self._fallback.detect(frame)
        if not direct_dets:
            return []

        best = max(direct_dets, key=lambda d: d.score)
        if best.score < self._cfg.direct_esp32_min_score:
            return []

        board_box = self._estimate_board_from_esp32(best.bbox, frame.shape)
        return [
            Detection(self._cfg.board_label, max(0.50, best.score - 0.05), board_box),
            Detection(self._cfg.esp32_label, best.score, self._clip_bbox(best.bbox, frame.shape)),
        ]

    def _detect_esp32_in_axis_aligned_crop(
        self,
        crop: np.ndarray,
        board_bbox: BBox,
        frame_shape: tuple[int, ...],
    ) -> list[Detection]:
        if crop.size == 0:
            return []

        out_w, out_h = self._cfg.board_cfg.output_size
        resized = cv.resize(crop, (out_w, out_h), interpolation=cv.INTER_AREA)
        warped_dets = self._esp32_in_board.detect(resized)
        if not warped_dets:
            return []

        board_w = max(1, board_bbox.x2 - board_bbox.x1)
        board_h = max(1, board_bbox.y2 - board_bbox.y1)
        sx = board_w / float(out_w)
        sy = board_h / float(out_h)

        mapped_dets: list[Detection] = []
        for det in warped_dets:
            if det.score < self._cfg.esp32_min_score_after_warp:
                continue
            box = det.bbox
            mapped = BBox(
                x1=board_bbox.x1 + int(round(box.x1 * sx)),
                y1=board_bbox.y1 + int(round(box.y1 * sy)),
                x2=board_bbox.x1 + int(round(box.x2 * sx)),
                y2=board_bbox.y1 + int(round(box.y2 * sy)),
            )
            mapped = self._clip_bbox(mapped, frame_shape)
            if mapped.area() <= 0:
                continue
            mapped_dets.append(Detection(self._cfg.esp32_label, det.score, mapped))

        return mapped_dets

    def _merge_board_and_esp32(
        self,
        board_first_dets: list[Detection],
        esp32_anchor_dets: list[Detection],
    ) -> list[Detection]:
        board_det = self._best_by_label(board_first_dets, self._cfg.board_label)
        esp32_det = self._best_by_label(esp32_anchor_dets, self._cfg.esp32_label)

        if board_det is None or esp32_det is None:
            return []

        if not self._bbox_contains_center(board_det.bbox, esp32_det.bbox):
            return []

        return [board_det, esp32_det]

    def _estimate_board_from_esp32(self, esp32_bbox: BBox, frame_shape: tuple[int, ...]) -> BBox:
        """
        Estimate the full board box from the ESP32 shield box.

        Ratios are approximate and tuned for the FireBeetle-style board:
        - board extends a bit left of the module
        - much more to the right
        - slightly above and below
        """
        h, w = frame_shape[:2]
        ew = max(1, esp32_bbox.x2 - esp32_bbox.x1)
        eh = max(1, esp32_bbox.y2 - esp32_bbox.y1)

        x1 = int(round(esp32_bbox.x1 - 0.30 * ew))
        y1 = int(round(esp32_bbox.y1 - 0.22 * eh))
        x2 = int(round(esp32_bbox.x2 + 1.85 * ew))
        y2 = int(round(esp32_bbox.y2 + 0.48 * eh))

        return self._clip_bbox(BBox(x1, y1, x2, y2), frame_shape)

    @staticmethod
    def _bbox_contains_center(outer: BBox, inner: BBox) -> bool:
        cx = 0.5 * (inner.x1 + inner.x2)
        cy = 0.5 * (inner.y1 + inner.y2)
        return outer.x1 <= cx <= outer.x2 and outer.y1 <= cy <= outer.y2

    @staticmethod
    def _best_by_label(dets: list[Detection], label: str) -> Detection | None:
        candidates = [d for d in dets if d.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.score)

    @staticmethod
    def _has_label(dets: list[Detection], label: str) -> bool:
        return any(d.label == label for d in dets)

    @staticmethod
    def _clip_bbox(bbox: BBox, frame_shape: tuple[int, ...]) -> BBox:
        h, w = frame_shape[:2]
        x1 = int(max(0, min(w - 1, bbox.x1)))
        y1 = int(max(0, min(h - 1, bbox.y1)))
        x2 = int(max(0, min(w - 1, bbox.x2)))
        y2 = int(max(0, min(h - 1, bbox.y2)))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return BBox(x1, y1, x2, y2)

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