# ORB - Fallback(optional)
# ORB-Fallback, wenn Templates bei Rotation/Belichtung schwächeln.
"""
feature_match.py

This module implements an optional ORB-based feature matching detector.
It can be used as a fallback when template matching becomes unstable under
rotation, viewpoint changes, or illumination changes.

Inputs:
- Scene image (preprocessed)
- Template image(s)
- ORB and matcher configuration parameters

Outputs:
- Estimated object location as BBox or list[Detection]
- Optional homography matrix for debugging

Zu implementierende Funktionen

    orb_detect_and_compute(image) -> (kps, desc)

    match_descriptors(desc1, desc2) -> matches

    estimate_homography(kps1, kps2, matches) -> H | None

    detect_via_homography(scene, template) -> BBox | None

ORB:
https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

Feature matching:
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

findHomography:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html


"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.detection_logic.postprocess import translate_detections
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class ORBMatchConfig:
    label: str = "ESP32"
    nfeatures: int = 1000
    good_match_ratio: float = 0.75
    min_inliers: int = 12
    search_roi: tuple[int, int, int, int] | None = None


class ORBFeatureMatcher(Detector):
    """ORB + BFMatcher + homography fallback detector."""

    def __init__(self, template_image: np.ndarray, cfg: ORBMatchConfig) -> None:
        self._cfg = cfg
        self._orb = cv.ORB_create(nfeatures=cfg.nfeatures)
        self._bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        self._tmpl_gray = self._ensure_gray(template_image)
        self._kp_t, self._des_t = self._orb.detectAndCompute(self._tmpl_gray, None)
        if self._des_t is None or len(self._kp_t) == 0:
            raise ValueError("ORB could not compute descriptors for the template")

        h, w = self._tmpl_gray.shape[:2]
        self._tmpl_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        search_img, dx, dy = self._crop_to_roi(frame)
        gray = self._ensure_gray(search_img)
        kp_f, des_f = self._orb.detectAndCompute(gray, None)
        if des_f is None or len(kp_f) < 8:
            return []

        matches = self._bf.knnMatch(self._des_t, des_f, k=2)
        good: list[cv.DMatch] = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self._cfg.good_match_ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            return []

        src_pts = np.float32([self._kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if H is None or mask is None:
            return []

        inliers = int(mask.sum())
        if inliers < self._cfg.min_inliers:
            return []

        projected = cv.perspectiveTransform(self._tmpl_corners, H).reshape(-1, 2)
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        det = Detection(
            label=self._cfg.label,
            score=float(min(1.0, inliers / max(1, len(good)))),
            bbox=BBox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
        )
        return translate_detections([det], dx, dy)

    def _crop_to_roi(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self._cfg.search_roi is None:
            return frame, 0, 0
        x, y, w, h = self._cfg.search_roi
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        x2 = max(x + 1, min(x + w, w_frame))
        y2 = max(y + 1, min(y + h, h_frame))
        return frame[y:y2, x:x2], x, y

    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image shape: {img.shape}")
