from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from src.utils.types import BBox, Detection
from src.detection_logic.template_match import TemplateMatcher

LOGGER = logging.getLogger(__name__)


@dataclass
class _BoardCandidate:
    quad: np.ndarray
    bbox: BBox
    geom_score: float
    verify_score: float
    combined_score: float
    H_img_to_board: np.ndarray
    H_board_to_img: np.ndarray
    board_warp: np.ndarray


class BoardFirstHybridDetector:
    """
    Experimental hybrid detector.
    Not used as the main detector right now because false positives on
    large rectangular scene structures are still too frequent.
    """

    CANON_W = 960
    CANON_H = 480
    IDEAL_ASPECT = 2.0

    def __init__(
        self,
        board_template_dir: Path,
        esp32_matcher: TemplateMatcher,
        source_mode: str = "images",
    ) -> None:
        self._board_template_dir = Path(board_template_dir)
        self._esp32_matcher = esp32_matcher
        self._source_mode = source_mode

        self._board_templates_gray, self._board_templates_edge = self._load_board_templates(
            self._board_template_dir,
            limit=12 if source_mode == "images" else 6,
        )

        self._last_board_bbox: Optional[BBox] = None
        self._last_seen_frames = 0

        self._board_min_score = 0.62 if source_mode == "images" else 0.72
        self._esp32_min_score = 0.62 if source_mode == "images" else 0.66

    def detect(self, frame: np.ndarray) -> List[Detection]:
        frame_h, frame_w = frame.shape[:2]
        candidates: List[_BoardCandidate] = []

        rois = [BBox(0, 0, frame_w, frame_h)]

        if self._source_mode == "live" and self._last_board_bbox is not None:
            rois.insert(0, self._expand_bbox(self._last_board_bbox, frame_w, frame_h, scale=1.5))

        seen = set()
        for roi in rois:
            for cand in self._find_candidates_in_roi(frame, roi):
                key = (cand.bbox.x1 // 8, cand.bbox.y1 // 8, cand.bbox.x2 // 8, cand.bbox.y2 // 8)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)

        if not candidates:
            LOGGER.debug("Hybrid detector: no candidates.")
            self._last_seen_frames = max(0, self._last_seen_frames - 1)
            if self._last_seen_frames == 0:
                self._last_board_bbox = None
            return []

        best = max(candidates, key=lambda c: c.combined_score)

        LOGGER.debug(
            "Hybrid detector: candidates=%d geom=%.3f verify=%.3f combined=%.3f bbox=(%d,%d,%d,%d)",
            len(candidates),
            best.geom_score,
            best.verify_score,
            best.combined_score,
            best.bbox.x1,
            best.bbox.y1,
            best.bbox.x2,
            best.bbox.y2,
        )

        if best.combined_score < self._board_min_score:
            return []

        self._last_board_bbox = best.bbox
        self._last_seen_frames = 3

        detections = [Detection("BOARD", float(best.combined_score), best.bbox)]

        esp32_dets = self._esp32_matcher.detect(best.board_warp)
        esp32_dets = [d for d in esp32_dets if d.score >= self._esp32_min_score]

        if not esp32_dets:
            return detections

        best_esp32 = max(esp32_dets, key=lambda d: d.score)
        mapped = self._map_bbox_from_board_to_image(best_esp32.bbox, best.H_board_to_img, frame_w, frame_h)
        detections.append(Detection("ESP32", float(best_esp32.score), mapped))
        return detections

    def _find_candidates_in_roi(self, frame: np.ndarray, roi: BBox) -> List[_BoardCandidate]:
        roi_img = frame[roi.y1:roi.y2, roi.x1:roi.x2]
        if roi_img.size == 0:
            return []

        gray = self._to_gray(roi_img)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blur, 60, 160)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = frame.shape[:2]
        roi_h, roi_w = gray.shape[:2]
        roi_area = float(max(1, roi_h * roi_w))

        raw_quads: List[Tuple[np.ndarray, float]] = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 0.01 * roi_area:
                continue
            if area > 0.45 * roi_area:
                continue

            rect = cv.minAreaRect(cnt)
            (cx, cy), (rw, rh), _ = rect

            if min(rw, rh) < 30:
                continue

            box = cv.boxPoints(rect).astype(np.float32)
            rect_area = max(rw * rh, 1.0)
            rectangularity = float(area / rect_area)
            if rectangularity < 0.55:
                continue

            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            aspect_score = max(0.0, 1.0 - abs(aspect - self.IDEAL_ASPECT) / 1.0)

            box[:, 0] += roi.x1
            box[:, 1] += roi.y1

            bbox = self._bbox_from_quad(box, frame_w, frame_h)
            bbox_area_ratio = self._bbox_area(bbox) / float(frame_w * frame_h)

            # Important rejection:
            # eliminate very large scene rectangles like window/wall regions
            if bbox_area_ratio > 0.35:
                continue

            touch_count = self._touching_border_count(bbox, frame_w, frame_h, margin=8)
            if touch_count >= 2:
                continue

            geom_score = 0.55 * rectangularity + 0.45 * aspect_score
            raw_quads.append((box, float(geom_score)))

        raw_quads.sort(key=lambda item: item[1], reverse=True)
        raw_quads = raw_quads[:6]

        dst = np.array(
            [
                [0, 0],
                [self.CANON_W - 1, 0],
                [self.CANON_W - 1, self.CANON_H - 1],
                [0, self.CANON_H - 1],
            ],
            dtype=np.float32,
        )

        out: List[_BoardCandidate] = []
        for quad, geom_score in raw_quads:
            ordered = self._order_quad_long_side_first(quad)
            H_img_to_board = cv.getPerspectiveTransform(ordered, dst)
            H_board_to_img = np.linalg.inv(H_img_to_board)
            board_warp = cv.warpPerspective(frame, H_img_to_board, (self.CANON_W, self.CANON_H))
            verify_score = self._board_verify_score(board_warp)
            combined = 0.50 * geom_score + 0.50 * verify_score
            bbox = self._bbox_from_quad(ordered, frame_w, frame_h)

            out.append(
                _BoardCandidate(
                    quad=ordered,
                    bbox=bbox,
                    geom_score=float(geom_score),
                    verify_score=float(verify_score),
                    combined_score=float(combined),
                    H_img_to_board=H_img_to_board,
                    H_board_to_img=H_board_to_img,
                    board_warp=board_warp,
                )
            )

        return out

    def _load_board_templates(self, template_dir: Path, limit: int):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            files.extend(sorted(template_dir.glob(ext)))

        gray_templates = []
        edge_templates = []

        for path in files[:limit]:
            img = cv.imread(str(path), cv.IMREAD_COLOR)
            if img is None:
                continue
            gray = self._prepare_board_score_image(img)
            edge = cv.Canny(gray, 60, 160)
            gray_templates.append(gray)
            edge_templates.append(edge)

        if not gray_templates:
            raise RuntimeError(f"No valid board templates found in: {template_dir}")

        return gray_templates, edge_templates

    def _board_verify_score(self, board_warp: np.ndarray) -> float:
        gray = self._prepare_board_score_image(board_warp)
        edge = cv.Canny(gray, 60, 160)

        best = -1.0
        for tmpl_gray, tmpl_edge in zip(self._board_templates_gray, self._board_templates_edge):
            gray_score = float(cv.matchTemplate(gray, tmpl_gray, cv.TM_CCOEFF_NORMED)[0, 0])
            edge_score = float(cv.matchTemplate(edge, tmpl_edge, cv.TM_CCOEFF_NORMED)[0, 0])
            score = 0.7 * gray_score + 0.3 * edge_score
            best = max(best, score)

        return float(best)

    def _prepare_board_score_image(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)
        gray = cv.resize(gray, (self.CANON_W, self.CANON_H), interpolation=cv.INTER_AREA)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return gray

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        else:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def _expand_bbox(bbox: BBox, frame_w: int, frame_h: int, scale: float) -> BBox:
        cx = 0.5 * (bbox.x1 + bbox.x2)
        cy = 0.5 * (bbox.y1 + bbox.y2)
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1

        nw = bw * scale
        nh = bh * scale

        x1 = int(max(0, round(cx - nw / 2)))
        y1 = int(max(0, round(cy - nh / 2)))
        x2 = int(min(frame_w, round(cx + nw / 2)))
        y2 = int(min(frame_h, round(cy + nh / 2)))
        return BBox(x1, y1, x2, y2)

    @staticmethod
    def _bbox_from_quad(quad: np.ndarray, frame_w: int, frame_h: int) -> BBox:
        x1 = int(max(0, np.floor(np.min(quad[:, 0]))))
        y1 = int(max(0, np.floor(np.min(quad[:, 1]))))
        x2 = int(min(frame_w - 1, np.ceil(np.max(quad[:, 0]))))
        y2 = int(min(frame_h - 1, np.ceil(np.max(quad[:, 1]))))
        return BBox(x1, y1, x2, y2)

    @staticmethod
    def _bbox_area(bbox: BBox) -> int:
        return max(0, bbox.x2 - bbox.x1) * max(0, bbox.y2 - bbox.y1)

    @staticmethod
    def _touching_border_count(bbox: BBox, frame_w: int, frame_h: int, margin: int = 6) -> int:
        count = 0
        if bbox.x1 <= margin:
            count += 1
        if bbox.y1 <= margin:
            count += 1
        if bbox.x2 >= frame_w - 1 - margin:
            count += 1
        if bbox.y2 >= frame_h - 1 - margin:
            count += 1
        return count

    @staticmethod
    def _order_quad_long_side_first(quad: np.ndarray) -> np.ndarray:
        pts = quad.reshape(4, 2).astype(np.float32)

        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        ordered = np.array([tl, tr, br, bl], dtype=np.float32)

        top = np.linalg.norm(ordered[1] - ordered[0])
        left = np.linalg.norm(ordered[3] - ordered[0])

        if left > top:
            ordered = np.array([ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32)

        return ordered

    @staticmethod
    def _map_bbox_from_board_to_image(
        bbox: BBox,
        H_board_to_img: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> BBox:
        corners = np.array(
            [
                [bbox.x1, bbox.y1],
                [bbox.x2, bbox.y1],
                [bbox.x2, bbox.y2],
                [bbox.x1, bbox.y2],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        projected = cv.perspectiveTransform(corners, H_board_to_img).reshape(-1, 2)

        x1 = int(max(0, np.floor(np.min(projected[:, 0]))))
        y1 = int(max(0, np.floor(np.min(projected[:, 1]))))
        x2 = int(min(frame_w - 1, np.ceil(np.max(projected[:, 0]))))
        y2 = int(min(frame_h - 1, np.ceil(np.max(projected[:, 1]))))

        return BBox(x1, y1, x2, y2)