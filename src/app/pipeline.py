# Orchestrierung: capture -> prepocess -> detect -> render

"""
pipeline.py

This module orchestrates the end-to-end processing pipeline:
capture -> (optional) board normalization -> preprocessing -> detection -> postprocess -> render.

The pipeline is designed for testability by using dependency injection:
- FrameSource provides frames (webcam or video file)
- preprocessors transform frames
- Detector returns structured detections
- Renderer draws overlays

Inputs:
- FrameSource instance
- Configuration dictionary
- Processing components (preprocessors, detector, renderer)

Outputs:
- Rendered frames (for display or debug saving)
- Logs describing runtime status and detection results
"""