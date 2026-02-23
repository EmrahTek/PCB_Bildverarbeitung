# dataclasses: BBox, Detection, FrameMeta

"""
types.py

This module defines core data structures used across the PCB component detection project.
The goal is to enforce clear, testable interfaces between the pipeline stages (capture,
preprocessing, detection, postprocessing, rendering).

Key concepts:
- BBox: Axis-aligned bounding box in pixel coordinates.
- Detection: A predicted object instance with label, confidence score, and BBox.
- FrameMeta: Metadata for a captured frame (frame_id, timestamp, source).

Inputs:
- No direct inputs (data classes only).

Outputs:
- Dataclass definitions that are imported by other modules.
"""