# Detector Interface

"""
base.py

This module defines the detector interface used by the pipeline.
All detection implementations must return a list of Detection objects.

Inputs:
- Preprocessed image (e.g., warped board image or ROI)

Outputs:
- list[Detection] containing label, confidence score, and bounding box
"""