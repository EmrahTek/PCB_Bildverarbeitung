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

# src/detection_logic/feature_match.py
from __future__ import annotations

from dataclasses import dataclass
import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class ORBMatchConfig:
    """
    ORB-based feature matching configuration (optional fallback).

    This is more robust to rotation/scale than template matching,
    but can be slower and needs textured templates.
    """
    label: str = "ESP32"
    nfeatures: int = 1000
    good_match_ratio: float = 0.75
    min_inliers: int = 12  # homography inliers threshold


class ORBFeatureMatcher(Detector):
    """
    ORB + BFMatcher + Homography detector.
    Returns a bounding box if homography is found with enough inliers.
    """

    def __init__(self, template_bgr: np.ndarray, cfg: ORBMatchConfig) -> None:
        self._cfg = cfg
        self._orb = cv.ORB_create(nfeatures=cfg.nfeatures)
        self._bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        self._tmpl_gray = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr
        self._kp_t, self._des_t = self._orb.detectAndCompute(self._tmpl_gray, None)
        if self._des_t is None:
            raise ValueError("ORB could not compute descriptors for template.")

        h, w = self._tmpl_gray.shape[:2]
        self._tmpl_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        kp_f, des_f = self._orb.detectAndCompute(gray, None)
        if des_f is None:
            return []

        matches = self._bf.knnMatch(self._des_t, des_f, k=2)

        # Lowe ratio test
        good = []
        for m, n in matches:
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

        # Project template corners to frame to get bbox
        proj = cv.perspectiveTransform(self._tmpl_corners, H).reshape(-1, 2)
        x1, y1 = proj.min(axis=0)
        x2, y2 = proj.max(axis=0)

        bbox = BBox(int(x1), int(y1), int(x2), int(y2))
        score = min(1.0, inliers / max(1, len(good)))  # heuristic confidence
        return [Detection(self._cfg.label, float(score), bbox)]

