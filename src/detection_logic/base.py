"""Shared detector interface for the pipeline.

Every detector implementation receives an OpenCV/NumPy frame and returns a list
of Detection objects with labels, confidence scores, and bounding boxes.

Python docs:
- abc: https://docs.python.org/3/library/abc.html
- typing: https://docs.python.org/3/library/typing.html
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.utils.types import Detection


class Detector(ABC):
    """
    Base interface for all detectors.

    A detector must implement:
        detect(frame) -> list[Detection]
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame."""
        raise NotImplementedError
