# Baseline: matchTemplate + Multi-Scale
# Baseline-Detektion per Template Matching (Multi-Scale), optional auf Gray oder Edges.

"""
template_match.py

This module implements template matching based detection using OpenCV's matchTemplate.
It supports:
- matching multiple templates per label
- multi-scale matching via image pyramids
- candidate extraction from score maps

Inputs:
- Preprocessed image (grayscale or edge map)
- Template images (grayscale or edge map)
- Matching configuration (method, thresholds, scales)

Outputs:
- list[Detection] with label, score, and bounding boxes

Zu implementierende Funktionen

    build_pyramid(image, scales) -> list[tuple[scale, image_scaled]]

    match_single_template(image, template, method) -> score_map

    extract_candidates(score_map, threshold) -> list[BBox+score]

    multi_scale_template_match(image, templates, scales, threshold) -> list[Detection]

    (Optional) prepare_image_for_matching(image, mode="gray|edges") -> img

matchTemplate tutorial:
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

matchTemplate reference:
https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html

Image pyramids (search terms):
"opencv image pyramid python"

"""
