from __future__ import annotations

import numpy as np

from src.preprocessing.geometry import order_quad_points


def test_order_quad_points_returns_consistent_order() -> None:
    pts = np.array([[300, 100], [100, 300], [100, 100], [300, 300]], dtype=np.float32)
    ordered = order_quad_points(pts)
    assert ordered.shape == (4, 2)
    assert np.allclose(ordered[0], [100, 100])
    assert np.allclose(ordered[1], [300, 100])
    assert np.allclose(ordered[2], [300, 300])
    assert np.allclose(ordered[3], [100, 300])
