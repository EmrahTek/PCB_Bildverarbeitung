from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.template_match import TemplateMatcher
from src.preprocessing.geometry import BoardWarpConfig, detect_and_warp_board
from src.utils.types import BBox, Detection
import logging
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelativeROI:
    x1f: float
    y1f: float
    x2f: float
    y2f: float


@dataclass(frozen=True)
class BoardFirstEsp32Config:
    board_cfg: BoardWarpConfig = BoardWarpConfig()
    board_label: str = "BOARD"
    esp32_label: str = "ESP32"
    usb_label: str = "USB_PORT"
    jst_label: str = "JST_CONNECTOR"
    reset_label: str = "RESET_BUTTON"
    fallback_direct_esp32: bool = True
    esp32_min_score_after_warp: float = 0.60
    usb_min_score_after_warp: float = 0.46
    jst_min_score_after_warp: float = 0.44
    reset_min_score_after_warp: float = 0.52
    direct_board_min_score: float = 0.56
    direct_esp32_min_score: float = 0.68


class BoardFirstEsp32Detector(Detector):
    """
    Final cascade detector:

    1) geometry -> warp board -> ROI component detection on canonical board
    2) direct board template fallback -> resize crop -> ROI component detection
    3) direct ESP32 fallback -> estimate board from ESP32 anchor
    """

    _ESP32_ROI = RelativeROI(0.03, 0.08, 0.50, 0.86)
    _USB_ROI   = RelativeROI(0.72, 0.03, 1.00, 0.46)
    _JST_ROI   = RelativeROI(0.70, 0.28, 1.00, 0.86)
    _RESET_ROI = RelativeROI(0.50, 0.10, 0.78, 0.48)

    def __init__(
        self,
        esp32_in_board_matcher: TemplateMatcher,
        cfg: BoardFirstEsp32Config = BoardFirstEsp32Config(),
        *,
        direct_fallback_matcher: TemplateMatcher | None = None,
        direct_board_matcher: TemplateMatcher | None = None,
        usb_in_board_matcher: TemplateMatcher | None = None,
        jst_in_board_matcher: TemplateMatcher | None = None,
        reset_in_board_matcher: TemplateMatcher | None = None,
    ) -> None:
        self._esp32_in_board = esp32_in_board_matcher
        self._fallback = direct_fallback_matcher
        self._board_full_frame = direct_board_matcher
        self._usb_in_board = usb_in_board_matcher
        self._jst_in_board = jst_in_board_matcher
        self._reset_in_board = reset_in_board_matcher
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
    # 1) First try geometry board + canonical ROI components
        geom_dets = self._detect_via_geometry(frame)
        if geom_dets:
        # If geometry found a board, trust that board first.
        # Do NOT run direct board fallback anymore.
        # Only return geometry-based results.
            return geom_dets

    # 2) Only if geometry failed completely, try direct board template fallback
        board_tpl_dets = self._detect_via_board_template(frame)
        if board_tpl_dets:
            return board_tpl_dets

    # 3) Last resort: direct ESP32 fallback
        direct_esp32_dets = self._detect_via_direct_esp32(frame)
        if direct_esp32_dets:
            return direct_esp32_dets

        return []

    def _detect_via_geometry(self, frame: np.ndarray) -> list[Detection]:
        result = detect_and_warp_board(frame, self._cfg.board_cfg)
        if result is None:
            return []

        board_bbox = self._quad_to_bbox(result.quad, frame.shape)
        detections: list[Detection] = [
            Detection(self._cfg.board_label, max(0.60, min(1.0, result.score + 0.15)), board_bbox)
        ]

        H_inv = np.linalg.inv(result.homography)
        detections.extend(
            self._detect_components_in_canonical(
                result.warped,
                map_bbox_fn=lambda box: self._map_bbox_back(box, H_inv, frame.shape),
            )
        )
        board_score = max(0.60, min(1.0, result.score + 0.15))
        if board_score < 0.58:
            return []
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
        detections: list[Detection] = [Detection(self._cfg.board_label, best_board.score, board_box)]

        crop = frame[board_box.y1:board_box.y2, board_box.x1:board_box.x2]
        if crop.size == 0:
            return detections

        out_w, out_h = self._cfg.board_cfg.output_size
        canonical = cv.resize(crop, (out_w, out_h), interpolation=cv.INTER_AREA)

        def _axis_map(box: BBox) -> BBox:
            board_w = max(1, board_box.x2 - board_box.x1)
            board_h = max(1, board_box.y2 - board_box.y1)
            sx = board_w / float(out_w)
            sy = board_h / float(out_h)
            mapped = BBox(
                x1=board_box.x1 + int(round(box.x1 * sx)),
                y1=board_box.y1 + int(round(box.y1 * sy)),
                x2=board_box.x1 + int(round(box.x2 * sx)),
                y2=board_box.y1 + int(round(box.y2 * sy)),
            )
            return self._clip_bbox(mapped, frame.shape)

        detections.extend(self._detect_components_in_canonical(canonical, map_bbox_fn=_axis_map))
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

    def _detect_components_in_canonical(self, canonical: np.ndarray, map_bbox_fn) -> list[Detection]:
        out: list[Detection] = []

        comp_specs = [
            (self._cfg.esp32_label, self._esp32_in_board, self._ESP32_ROI, self._cfg.esp32_min_score_after_warp),
            (self._cfg.usb_label, self._usb_in_board, self._USB_ROI, self._cfg.usb_min_score_after_warp),
            (self._cfg.jst_label, self._jst_in_board, self._JST_ROI, self._cfg.jst_min_score_after_warp),
            (self._cfg.reset_label, self._reset_in_board, self._RESET_ROI, self._cfg.reset_min_score_after_warp),
        ]

        for label, matcher, roi, min_score in comp_specs:
            if matcher is None:
                continue
            det = self._detect_single_component_in_roi(canonical, matcher, label, roi, min_score)
            if det is None:
                continue
            mapped_bbox = map_bbox_fn(det.bbox)
            if mapped_bbox is None or mapped_bbox.area() <= 0:
                continue
            out.append(Detection(label, det.score, mapped_bbox))

        return out

    def _detect_single_component_in_roi(
        self,
        canonical: np.ndarray,
        matcher: TemplateMatcher,
        label: str,
        roi: RelativeROI,
        min_score: float,
    ) -> Detection | None:
        roi_box = self._relative_roi_to_bbox(roi, canonical.shape)
        crop = canonical[roi_box.y1:roi_box.y2, roi_box.x1:roi_box.x2]
        if crop.size == 0:
            LOGGER.debug(
                "ROI component %s: empty crop roi=(%d,%d,%d,%d)",
                label,
                roi_box.x1,
                roi_box.y1,
                roi_box.x2,
                roi_box.y2,
            )
            return None

        raw_best = matcher.best_raw_score(crop)
        dets = matcher.detect(crop)

        if not dets:
            LOGGER.debug(
                "ROI component %s: no detections raw_best=%.3f threshold=%.3f roi=(%d,%d,%d,%d)",
                label,
                raw_best,
                min_score,
                roi_box.x1,
                roi_box.y1,
                roi_box.x2,
                roi_box.y2,
            )
            return None

        best = max(dets, key=lambda d: d.score)

        LOGGER.debug(
            "ROI component %s: best_score=%.3f raw_best=%.3f threshold=%.3f roi=(%d,%d,%d,%d) num_candidates=%d",
            label,
            best.score,
            raw_best,
            min_score,
            roi_box.x1,
            roi_box.y1,
            roi_box.x2,
            roi_box.y2,
            len(dets),
        )

        if best.score < min_score:
            return None

        mapped = BBox(
            x1=roi_box.x1 + best.bbox.x1,
            y1=roi_box.y1 + best.bbox.y1,
            x2=roi_box.x1 + best.bbox.x2,
            y2=roi_box.y1 + best.bbox.y2,
        )
        return Detection(label, best.score, mapped)

    def _merge_board_and_esp32(self, board_first_dets: list[Detection], esp32_anchor_dets: list[Detection]) -> list[Detection]:
        board_det = self._best_by_label(board_first_dets, self._cfg.board_label)
        esp32_det = self._best_by_label(esp32_anchor_dets, self._cfg.esp32_label)
        if board_det is None or esp32_det is None:
            return []
        if not self._bbox_contains_center(board_det.bbox, esp32_det.bbox):
            return []

        merged = [board_det, esp32_det]
        for label in (self._cfg.usb_label, self._cfg.jst_label, self._cfg.reset_label):
            comp = self._best_by_label(board_first_dets, label)
            if comp is not None:
                merged.append(comp)
        return merged

    def _estimate_board_from_esp32(self, esp32_bbox: BBox, frame_shape: tuple[int, ...]) -> BBox:
        ew = max(1, esp32_bbox.x2 - esp32_bbox.x1)
        eh = max(1, esp32_bbox.y2 - esp32_bbox.y1)

        x1 = int(round(esp32_bbox.x1 - 0.30 * ew))
        y1 = int(round(esp32_bbox.y1 - 0.22 * eh))
        x2 = int(round(esp32_bbox.x2 + 1.85 * ew))
        y2 = int(round(esp32_bbox.y2 + 0.48 * eh))

        return self._clip_bbox(BBox(x1, y1, x2, y2), frame_shape)

    @staticmethod
    def _relative_roi_to_bbox(roi: RelativeROI, frame_shape: tuple[int, ...]) -> BBox:
        h, w = frame_shape[:2]
        x1 = int(round(roi.x1f * w))
        y1 = int(round(roi.y1f * h))
        x2 = int(round(roi.x2f * w))
        y2 = int(round(roi.y2f * h))
        return BBox(x1, y1, x2, y2)

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