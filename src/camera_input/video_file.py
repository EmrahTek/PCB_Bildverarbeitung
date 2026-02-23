# deterministische Tests mit Video-Datei

"""
video_file.py

This module implements a deterministic FrameSource backed by a video file.
It is critical for reproducible testing and debugging, allowing the pipeline
to run without a live camera.

Inputs:
- Path to a video file
- loop flag to replay the video

Outputs:
- Frames as NumPy arrays (BGR) and FrameMeta objects
"""