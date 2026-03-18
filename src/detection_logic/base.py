# Detector Interface
# Gemeinsames Interface für Detektoren (Template Matching jetzt; ORB optional später).

"""
base.py

This module defines the detector interface used by the pipeline.
All detection implementations must return a list of Detection objects.

Inputs:
- Preprocessed image (e.g., warped board image or ROI)

Outputs:
- list[Detection] containing label, confidence score, and bounding box

Zu implementierende Funktionen / Klassen

    class Detector(Protocol/ABC):

    detect(image: np.ndarray) -> list[Detection]

Same resources as base interfaces:
https://docs.python.org/3/library/abc.html
https://docs.python.org/3/library/typing.html#typing.Protocol

"""

# src/detection_logic/base.py
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
    def detect(self,frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame."""
        raise NotImplementedError