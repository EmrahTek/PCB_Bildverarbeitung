# Farbraum-Konvertierungen und optionale Normalisierung (robuster gegen Lichtwechsel).


"""
color.py

This module contains color space conversions and light normalization helpers.
Its main goal is to provide stable representations (e.g., grayscale, HSV)
for downstream algorithms (edge detection, template matching, etc.).

Inputs:
- BGR images as NumPy arrays (OpenCV default format)

Outputs:
- Grayscale or HSV images as NumPy arrays
- Optionally normalized images to reduce lighting sensitivity

Zu implementierende Funktionen

    to_gray(bgr: np.ndarray) -> np.ndarray

    to_hsv(bgr: np.ndarray) -> np.ndarray

    normalize_intensity(gray: np.ndarray) -> np.ndarray (optional)

    (Optional) white_balance_grayworld(bgr) -> bgr

cvtColor:
https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
"""

from __future__ import annotations

import cv2 as cv
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR or gray input to grayscale."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert BGR input to HSV."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("HSV conversion requires a BGR image")
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize grayscale images into uint8 range [0, 255]."""
    if gray.dtype == np.uint8:
        return gray
    return cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
