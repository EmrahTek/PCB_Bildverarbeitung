# ORB - Fallback(optional)

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
"""

