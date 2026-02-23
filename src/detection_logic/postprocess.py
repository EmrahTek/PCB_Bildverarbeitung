# NMS, Thresholding, Score-Filter

"""
postprocess.py

This module contains postprocessing utilities for raw detections:
- score filtering
- Non-Maximum Suppression (NMS) using IoU
- counting detections per label

Inputs:
- list[Detection] from a detector

Outputs:
- list[Detection] after filtering/NMS
- dict[label, count] for UI overlay and logging
"""

