# Baseline: matchTemplate + Multi-Scale

"""
template_match.py

This module implements template matching based detection using OpenCV's matchTemplate.
It supports:
- matching multiple templates per label
- multi-scale matching via image pyramids
- candidate extraction from score maps

Inputs:
- Preprocessed image (grayscale or edge map)
- Template images (grayscale or edge map)
- Matching configuration (method, thresholds, scales)

Outputs:
- list[Detection] with label, score, and bounding boxes
"""
