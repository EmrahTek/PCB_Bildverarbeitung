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