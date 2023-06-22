import numpy as np
import pytest

import flowops


def test_flow_init():
    flow_x = np.random.randn(10, 10)
    flow_y = np.random.randn(10, 10)

    flow = flowops.Flow(flow_x, flow_y)

    assert flow is not None
