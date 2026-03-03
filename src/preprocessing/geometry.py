# PCB finden (Konturen), Ecken ordnen, Homographie berechnen, perspektivisch auf Standardansicht warpen.


"""
geometry.py

This module implements board localization and perspective normalization.
The idea is to find the PCB in the camera frame, estimate a homography, and warp
the view into a canonical "top-down" board coordinate system.

This step greatly improves detection robustness by stabilizing scale and orientation
across frames.

Inputs:
- BGR frame (NumPy array)
- Configuration parameters (thresholds, target output size)

Outputs:
- warped_board: normalized board image (NumPy array)
- debug_info: optional metadata for visualization (corners, contour, homography)

"""


"""
Zu implementierende Funktionen

    find_board_contour(gray_or_edges) -> contour | None

    approx_quad(contour) -> np.ndarray[(4,2)] | None

    order_points(pts4) -> pts4_ordered

    compute_homography(src_pts, dst_pts) -> H

    warp_perspective(frame, H, out_size=(W,H)) -> warped

    detect_and_warp_board(frame, config) -> (warped, debug_info)

Contours:
https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

findContours:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

approxPolyDP:
https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html

Homography / warpPerspective:
https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html    

"""