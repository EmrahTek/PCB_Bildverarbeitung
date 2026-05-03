from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.postprocess import iou
from src.detection_logic.template_match import TemplateMatcher
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class BoardTemplateLocatorConfig:
    """Settings for coarse full-frame board template localisation."""
    resize_width: int = 960
    min_score: float = 0.28
    max_candidates: int = 8


class BoardTemplateLocator:
    """
    Lightweight full-frame board template matcher.

    It returns a coarse board bounding box that can be used as a hint for the
    geometry-based localizer. This is especially helpful when the board is
    attached to a stick or shown in front of cluttered backgrounds.
    """

    def __init__(self, matcher: TemplateMatcher, cfg: BoardTemplateLocatorConfig = BoardTemplateLocatorConfig()) -> None:
        self._matcher = matcher
        self._cfg = cfg

    def detect(self, frame: np.ndarray) -> Detection | None:
        candidates = self.detect_candidates(frame)
        return candidates[0] if candidates else None

    def detect_candidates(self, frame: np.ndarray) -> list[Detection]:
        """Return coarse board hints from template matching and line geometry."""
        search_frame, scale = self._resize_for_search(frame)
        candidates: list[Detection] = []
        template_candidates = self._matcher.detect_candidates(
            search_frame,
            max_candidates=max(1, self._cfg.max_candidates * 2),
            score_threshold=max(0.18, self._cfg.min_score * 0.85),
        )
        candidates.extend(self._scale_detection_back(detection, scale) for detection in template_candidates)

        line_candidates = self._line_pair_candidates(search_frame, scale)
        candidates.extend(line_candidates)
        min_hint_score = max(0.20, self._cfg.min_score * 0.80)
        candidates = [candidate for candidate in candidates if candidate.score >= min_hint_score]
        return self._nms_candidates(candidates)

    def _resize_for_search(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame.shape[:2]
        target_width = self._cfg.resize_width
        if target_width <= 0 or width <= target_width:
            return frame, 1.0

        scale = target_width / float(width)
        resized = cv.resize(frame, (target_width, int(round(height * scale))), interpolation=cv.INTER_AREA)
        return resized, scale

    def _scale_detection_back(self, detection: Detection, scale: float) -> Detection:
        if scale == 1.0:
            return detection
        return Detection(
            label=detection.label,
            score=detection.score,
            bbox=BBox(
                x1=int(round(detection.bbox.x1 / scale)),
                y1=int(round(detection.bbox.y1 / scale)),
                x2=int(round(detection.bbox.x2 / scale)),
                y2=int(round(detection.bbox.y2 / scale)),
            ),
        )

    def _line_pair_candidates(self, frame: np.ndarray, scale: float) -> list[Detection]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        edges = cv.Canny(gray, 45, 135)
        lines = cv.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=45,
            minLineLength=max(45, int(0.08 * frame.shape[1])),
            maxLineGap=18,
        )
        if lines is None:
            return []

        horizontal: list[tuple[int, int, int, int, float]] = []
        for raw in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in raw)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle > 90:
                angle = 180 - angle
            if angle > 12:
                continue
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            horizontal.append((x1, y1, x2, y2, float(np.hypot(dx, dy))))
        horizontal.sort(key=lambda item: item[4], reverse=True)
        horizontal = horizontal[:36]

        candidates: list[Detection] = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = max(1, frame_h * frame_w)
        for index, first in enumerate(horizontal):
            for second in horizontal[index + 1 :]:
                x1a, y1a, x2a, y2a, len_a = first
                x1b, y1b, x2b, y2b, len_b = second
                ya = 0.5 * (y1a + y2a)
                yb = 0.5 * (y1b + y2b)
                height = abs(yb - ya)
                if height < 18 or height > 0.40 * frame_h:
                    continue

                overlap = min(x2a, x2b) - max(x1a, x1b)
                if overlap < max(35, 0.25 * min(len_a, len_b)):
                    continue

                x1 = min(x1a, x1b)
                x2 = max(x2a, x2b)
                width = x2 - x1
                if width <= 0:
                    continue
                aspect = width / max(1.0, height)
                if aspect < 1.30 or aspect > 5.20:
                    continue

                margin_y = 0.28 * height
                margin_x = 0.04 * width
                bx1 = int(max(0, np.floor(x1 - margin_x)))
                bx2 = int(min(frame_w - 1, np.ceil(x2 + margin_x)))
                by1 = int(max(0, np.floor(min(ya, yb) - margin_y)))
                by2 = int(min(frame_h - 1, np.ceil(max(ya, yb) + margin_y)))
                bbox = BBox(bx1, by1, bx2, by2)
                area_ratio = bbox.area() / frame_area
                if area_ratio < 0.008 or area_ratio > 0.55:
                    continue

                aspect_score = float(np.exp(-abs(np.log(aspect / 2.0))))
                line_score = float(np.clip(0.26 + 0.30 * aspect_score + 0.20 * min(len_a, len_b) / max(1, frame_w), 0.20, 0.72))
                if scale != 1.0:
                    bbox = BBox(
                        x1=int(round(bbox.x1 / scale)),
                        y1=int(round(bbox.y1 / scale)),
                        x2=int(round(bbox.x2 / scale)),
                        y2=int(round(bbox.y2 / scale)),
                    )
                candidates.append(Detection(label="BOARD", score=line_score, bbox=bbox))

        candidates.sort(key=lambda det: det.score, reverse=True)
        return candidates[: self._cfg.max_candidates]

    def _nms_candidates(self, candidates: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for candidate in sorted(candidates, key=lambda det: det.score, reverse=True):
            if any(iou(candidate.bbox, existing.bbox) > 0.45 for existing in kept):
                continue
            kept.append(candidate)
            if len(kept) >= self._cfg.max_candidates:
                break
        return kept
