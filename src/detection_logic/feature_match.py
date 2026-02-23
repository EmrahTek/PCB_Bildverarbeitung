# ORB - Fallback(optional)
# ORB-Fallback, wenn Templates bei Rotation/Belichtung schwächeln.
"""
feature_match.py

This module implements an optional ORB-based feature matching detector.
It can be used as a fallback when template matching becomes unstable under
rotation, viewpoint changes, or illumination changes.

Inputs:
- Scene image (preprocessed)
- Template image(s)
- ORB and matcher configuration parameters

Outputs:
- Estimated object location as BBox or list[Detection]
- Optional homography matrix for debugging

Zu implementierende Funktionen

    orb_detect_and_compute(image) -> (kps, desc)

    match_descriptors(desc1, desc2) -> matches

    estimate_homography(kps1, kps2, matches) -> H | None

    detect_via_homography(scene, template) -> BBox | None

ORB:
https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

Feature matching:
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

findHomography:
https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html


"""

