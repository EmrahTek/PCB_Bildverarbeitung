# deterministische Tests mit Video-Datei
# Deterministische Frame-Quelle aus Video-Datei (Pflicht für Tests & Reproduzierbarkeit).
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

"""
Zu implementierende Funktionen / Klassen

class VideoFileSource(FrameSource):

    __init__(path: Path, loop: bool = False)

    open()

    read() (end-of-file handling)

    close()

    OpenCV VideoCapture also supports video files:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

"""