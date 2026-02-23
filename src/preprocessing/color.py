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
"""