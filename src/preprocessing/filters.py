"""
filters.py

This module provides reusable image filtering primitives such as:
- Gaussian blur
- CLAHE contrast enhancement
- Canny edge detection
- Optional morphological operators

Inputs:
- Grayscale or BGR images as NumPy arrays

Outputs:
- Filtered images (grayscale, edges, binary masks) as NumPy arrays
"""