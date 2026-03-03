# cv2.VideoCapture
# Webcam-Quelle via cv2.VideoCapture, mit stabiler Fehlerbehandlung und einstellbarer Auflösung.

"""
webcam.py

This module implements a FrameSource for a live webcam using cv2.VideoCapture.
It handles:
- opening the device
- configuring capture properties (resolution, fps)
- reading frames with robust error handling
- closing the device cleanly

Inputs:
- camera_index (int)
- desired width/height (int)
- optional fps (int)

Outputs:
- Frames as NumPy arrays (BGR) and FrameMeta objects

"""

"""
Zu implementierende Funktionen / Klassen

class WebcamSource(FrameSource):

    __init__(camera_index: int, width: int, height: int, fps: int | None)

    open()

    read()

    close()

    set_capture_properties(cap, width, height, fps) -> None

OpenCV VideoCapture:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

VideoCapture properties (CAP_PROP_*):
https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html

"""