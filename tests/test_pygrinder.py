"""
PyGrinder test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import numpy as np

from pygrinder import mcar

DEFAULT_MISSING_RATE = 0.1
NAN = 1


class TestPyGrinder(unittest.TestCase):
    def test_0_mcar(self):
        d = np.random.randn(128, 10, 36)
        d_intact, d_with_missing, missing_mask, indicating_mask = mcar(
            d, p=DEFAULT_MISSING_RATE, nan=NAN
        )
        shape_product = np.product(d.shape)
        actual_missing_rate = (shape_product - missing_mask.sum()) / shape_product
        assert (
            round(actual_missing_rate, 1) == DEFAULT_MISSING_RATE
        ), f"Actual missing rate is {actual_missing_rate}, not given {DEFAULT_MISSING_RATE}"
        assert np.sum(d_with_missing[(1 - missing_mask).astype(bool)]) == NAN * np.sum(
            1 - missing_mask
        )
