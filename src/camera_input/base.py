"""Shared interface for all frame sources.

A FrameSource hides whether frames come from a webcam, IDS camera, Pi camera,
video file, single image, or image folder, which keeps the pipeline testable.

Python docs:
- abc: https://docs.python.org/3/library/abc.html
- typing: https://docs.python.org/3/library/typing.html
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from src.utils.types import FrameMeta


class FrameSource(ABC):
    """Abstract frame source used by the runtime pipeline."""

    @abstractmethod
    def open(self) -> None:
        """Open the source (allocate resources)."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        """
        Read one frame.

        Returns:
            (frame, meta) or (None, None) when the stream ends / fails.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release resources (close capture)."""
        raise NotImplementedError
