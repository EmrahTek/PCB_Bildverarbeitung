from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

# KEEP your existing imports for BBox / Detection / TemplateMatcher
# Example:
from src.utils.types import BBox, Detection
from src.detection_logic.template_match import TemplateMatcher


@dataclass
class _BoardCandidate:
    quad: np.ndarray              # shape=(4,2), float32 in original image coords
    bbox: "BBox"
    geom_score: float
    verify_score: float
    combined_score: float
    H_img_to_board: np.ndarray
    H_board_to_img: np.ndarray
    board_warp: np.ndarray


class BoardFirstHybridDetector:
    """
    Board-first detector:
    1) find board candidates geometrically
    2) verify them with real board templates
    3) only then search ESP32 inside the warped board

    This is intentionally stricter for webcam/video than for image batches.
    """

    CANON_W = 960
    CANON_H = 480
    IDEAL_ASPECT = 2.0

    def __init__(
        self,
        board_template_dir: Path,
        esp32_matcher: "TemplateMatcher",
        source_mode: str = "images",  # "images" or "live"
    ) -> None:
        self._board_template_dir = Path(board_template_dir)
        self._esp32_matcher = esp32_matcher
        self._source_mode = source_mode

        self._board_templates_gray, self._board_templates_edge = self._load_board_templates(
            self._board_template_dir,
            limit=8 if source_mode == "images" else 4,
        )

        self._last_board_bbox: Optional["BBox"] = None
        self._last_seen_frames: int = 0

        self._board_min_score = 0.62 if source_mode == "images" else 0.74
        self._esp32_min_score = 0.66 if source_mode == "images" else 0.68

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List["Detection"]:
        frame_h, frame_w = frame.shape[:2]
        candidates: List[_BoardCandidate] = []

        rois = [self._full_bbox(frame_w, frame_h)]

        # In live mode, search first around the last good board position.
        if self._source_mode == "live" and self._last_board_bbox is not None:
            rois.insert(0, self._expand_bbox(self._last_board_bbox, frame_w, frame_h, scale=1.6))

        seen_keys = set()

        for roi in rois:
            for cand in self._find_candidates_in_roi(frame, roi):
                key = (
                    int(cand.bbox.x1 // 8),
                    int(cand.bbox.y1 // 8),
                    int(cand.bbox.x2 // 8),
                    int(cand.bbox.y2 // 8),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(cand)

        if not candidates:
            self._last_seen_frames = max(0, self._last_seen_frames - 1)
            if self._last_seen_frames == 0:
                self._last_board_bbox = None
            return []

        best = max(candidates, key=lambda c: c.combined_score)

        if best.combined_score < self._board_min_score:
            self._last_seen_frames = max(0, self._last_seen_frames - 1)
            if self._last_seen_frames == 0:
                self._last_board_bbox = None
            return []

        self._last_board_bbox = best.bbox
        self._last_seen_frames = 3

        detections: List["Detection"] = [
            Detection(label="BOARD", score=float(best.combined_score), bbox=best.bbox)
        ]

        # Blur gate: if warped board is too blurry, do not trust ESP32 result.
        board_gray = self._to_gray(best.board_warp)
        lap_var = cv.Laplacian(board_gray, cv.CV_64F).var()

        board_area_ratio = self._bbox_area(best.bbox) / float(frame_w * frame_h)
        if self._source_mode == "live":
            if lap_var < 28.0 or board_area_ratio < 0.05:
                return detections
        else:
            if lap_var < 18.0:
                return detections

        esp32_dets = self._esp32_matcher.detect(best.board_warp)
        esp32_dets = [d for d in esp32_dets if float(d.score) >= self._esp32_min_score]

        if not esp32_dets:
            return detections

        best_esp32 = max(esp32_dets, key=lambda d: float(d.score))
        mapped_bbox = self._map_bbox_from_board_to_image(
            best_esp32.bbox,
            best.H_board_to_img,
            frame_w,
            frame_h,
        )

        detections.append(
            Detection(label="ESP32", score=float(best_esp32.score), bbox=mapped_bbox)
        )
        return detections

    # ------------------------------------------------------------------
    # candidate search
    # ------------------------------------------------------------------
    def _find_candidates_in_roi(self, frame: np.ndarray, roi: "BBox") -> List[_BoardCandidate]:
        roi_img = frame[roi.y1:roi.y2, roi.x1:roi.x2]
        if roi_img.size == 0:
            return []

        gray = self._to_gray(roi_img)
        small_gray, scale = self._resize_for_board_search(gray)

        blur = cv.GaussianBlur(small_gray, (5, 5), 0)
        edges = cv.Canny(blur, 60, 160)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        _, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        inv_otsu = 255 - otsu
        inv_otsu = cv.morphologyEx(inv_otsu, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        masks = [edges, inv_otsu]
        raw_quads: List[Tuple[np.ndarray, float]] = []

        search_h, search_w = small_gray.shape[:2]
        search_area = float(search_w * search_h)

        for mask in masks:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < 0.02 * search_area:
                    continue
                if area > 0.92 * search_area:
                    continue

                rect = cv.minAreaRect(cnt)
                (cx, cy), (rw, rh), _ = rect

                if min(rw, rh) < 40:
                    continue

                box = cv.boxPoints(rect).astype(np.float32)
                rect_area = max(rw * rh, 1.0)
                rectangularity = float(area / rect_area)
                if rectangularity < 0.55:
                    continue

                aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
                aspect_score = max(0.0, 1.0 - abs(aspect - self.IDEAL_ASPECT) / 1.2)

                area_ratio = area / search_area

                # live mode: reject "whole frame" and huge wall/window candidates
                if self._source_mode == "live":
                    if area_ratio > 0.80:
                        continue

                center_score = self._center_score(cx, cy, search_w, search_h)

                geom_score = (
                    0.45 * rectangularity +
                    0.35 * aspect_score +
                    0.20 * center_score
                )

                # back to original frame coords
                box /= scale
                box[:, 0] += roi.x1
                box[:, 1] += roi.y1

                raw_quads.append((box.astype(np.float32), float(geom_score)))

        if not raw_quads:
            return []

        # Keep only strongest few candidates for template verification
        raw_quads.sort(key=lambda item: item[1], reverse=True)
        raw_quads = raw_quads[:6]

        candidates: List[_BoardCandidate] = []
        frame_h, frame_w = frame.shape[:2]

        for quad, geom_score in raw_quads:
            ordered = self._order_quad_long_side_first(quad)
            H_img_to_board = cv.getPerspectiveTransform(
                ordered,
                np.array(
                    [
                        [0, 0],
                        [self.CANON_W - 1, 0],
                        [self.CANON_W - 1, self.CANON_H - 1],
                        [0, self.CANON_H - 1],
                    ],
                    dtype=np.float32,
                ),
            )
            H_board_to_img = np.linalg.inv(H_img_to_board)
            board_warp = cv.warpPerspective(frame, H_img_to_board, (self.CANON_W, self.CANON_H))

            verify_score = self._board_verify_score(board_warp)
            combined = 0.42 * geom_score + 0.58 * verify_score

            bbox = self._bbox_from_quad(ordered, frame_w, frame_h)

            # Extra live rejection:
            # avoid very border-touching random rectangles unless score is very strong
            if self._source_mode == "live":
                touch_count = self._touching_border_count(bbox, frame_w, frame_h)
                if touch_count >= 2 and combined < 0.88:
                    continue

            candidates.append(
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

        return candidates

    # ------------------------------------------------------------------
    # board verification
    # ------------------------------------------------------------------
    def _load_board_templates(
        self,
        template_dir: Path,
        limit: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
            files.extend(sorted(template_dir.glob(ext)))

        gray_templates: List[np.ndarray] = []
        edge_templates: List[np.ndarray] = []

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

            score = 0.68 * gray_score + 0.32 * edge_score
            if score > best:
                best = score

        return float(best)

    def _prepare_board_score_image(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)
        gray = cv.resize(gray, (self.CANON_W, self.CANON_H), interpolation=cv.INTER_AREA)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return gray

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _resize_for_board_search(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = gray.shape[:2]
        max_dim = 960 if self._source_mode == "images" else 640
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max(h, w) / float(max_dim)
            new_w = max(1, int(round(w / scale)))
            new_h = max(1, int(round(h / scale)))
            gray = cv.resize(gray, (new_w, new_h), interpolation=cv.INTER_AREA)
        return gray, scale

    def _order_quad_long_side_first(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32)

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        ordered = np.array([tl, tr, br, bl], dtype=np.float32)

        top_len = np.linalg.norm(ordered[1] - ordered[0])
        right_len = np.linalg.norm(ordered[2] - ordered[1])

        # If vertical side is longer, rotate ordering so long side maps to width.
        if right_len > top_len:
            ordered = np.array([ordered[1], ordered[2], ordered[3], ordered[0]], dtype=np.float32)

        return ordered

    def _center_score(self, cx: float, cy: float, w: int, h: int) -> float:
        nx = (cx - 0.5 * w) / max(0.5 * w, 1.0)
        ny = (cy - 0.5 * h) / max(0.5 * h, 1.0)
        d = np.sqrt(nx * nx + ny * ny)
        return float(max(0.0, 1.0 - d))

    # ------------------------------------------------------------------
    # bbox helpers
    # ------------------------------------------------------------------
    def _full_bbox(self, frame_w: int, frame_h: int) -> "BBox":
        return BBox(0, 0, frame_w, frame_h)

    def _expand_bbox(self, bbox: "BBox", frame_w: int, frame_h: int, scale: float) -> "BBox":
        cx = 0.5 * (bbox.x1 + bbox.x2)
        cy = 0.5 * (bbox.y1 + bbox.y2)
        bw = (bbox.x2 - bbox.x1) * scale
        bh = (bbox.y2 - bbox.y1) * scale

        x1 = int(max(0, round(cx - bw / 2)))
        y1 = int(max(0, round(cy - bh / 2)))
        x2 = int(min(frame_w, round(cx + bw / 2)))
        y2 = int(min(frame_h, round(cy + bh / 2)))
        return BBox(x1, y1, x2, y2)

    def _bbox_from_quad(self, quad: np.ndarray, frame_w: int, frame_h: int) -> "BBox":
        x1 = int(max(0, np.floor(np.min(quad[:, 0]))))
        y1 = int(max(0, np.floor(np.min(quad[:, 1]))))
        x2 = int(min(frame_w, np.ceil(np.max(quad[:, 0]))))
        y2 = int(min(frame_h, np.ceil(np.max(quad[:, 1]))))
        return BBox(x1, y1, x2, y2)

    def _bbox_area(self, bbox: "BBox") -> int:
        return max(0, int(bbox.x2 - bbox.x1)) * max(0, int(bbox.y2 - bbox.y1))

    def _touching_border_count(self, bbox: "BBox", frame_w: int, frame_h: int) -> int:
        eps = 8
        count = 0
        if bbox.x1 <= eps:
            count += 1
        if bbox.y1 <= eps:
            count += 1
        if bbox.x2 >= frame_w - eps:
            count += 1
        if bbox.y2 >= frame_h - eps:
            count += 1
        return count

    # ------------------------------------------------------------------
    # mapping helpers
    # ------------------------------------------------------------------
    def _map_bbox_from_board_to_image(
        self,
        bbox: "BBox",
        H_board_to_img: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> "BBox":
        pts = np.array(
            [
                [bbox.x1, bbox.y1],
                [bbox.x2, bbox.y1],
                [bbox.x2, bbox.y2],
                [bbox.x1, bbox.y2],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        mapped = cv.perspectiveTransform(pts, H_board_to_img).reshape(-1, 2)
        x1 = int(max(0, np.floor(np.min(mapped[:, 0]))))
        y1 = int(max(0, np.floor(np.min(mapped[:, 1]))))
        x2 = int(min(frame_w, np.ceil(np.max(mapped[:, 0]))))
        y2 = int(min(frame_h, np.ceil(np.max(mapped[:, 1]))))
        return BBox(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)