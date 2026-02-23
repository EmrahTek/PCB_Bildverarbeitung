# cv2.VideoCapture

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