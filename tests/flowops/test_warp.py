import numpy as np
import pytest

import flowops


def test_warp_forward():
    flow_x = np.random.randn(10, 10) * 10
    flow_y = np.random.randn(10, 10) * 10
    image = np.random.rand(10, 10)

    warped = flowops.warp_forward(image, (flow_x, flow_y))

    assert warped.shape == image.shape


def test_warp_backward():
    flow_x = np.random.randn(10, 10) * 10
    flow_y = np.random.randn(10, 10) * 10
    image = np.random.rand(10, 10)

    warped = flowops.warp_forward(image, (flow_x, flow_y))

    assert warped.shape == image.shape
    ...
