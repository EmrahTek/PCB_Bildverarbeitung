# Boxes/Labels/Counts zeichnen
# Visualisierung: Bounding Boxes, Labels, Score, Count-Panel; Debug-Overlays (Board-Ecken).

"""
overlay.py

This module renders visualization overlays onto frames:
- bounding boxes and labels for detections
- confidence scores
- a counts panel showing how many instances per label were found
- optional debug overlay for board detection (corners/contours)

Inputs:
- Original or warped frame (NumPy array)
- list[Detection]
- counts dictionary (label -> int)
- optional debug_info (board corners, etc.)

Outputs:
- Annotated frame (NumPy array)

Zu implementierende Funktionen

    draw_detections(frame, detections) -> frame

    draw_counts_panel(frame, counts) -> frame

    draw_board_debug(frame, debug_info) -> frame

    put_fps(frame, fps_value) -> frame (oder via fps.py)





OpenCV drawing (rectangle, putText):
https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
"""