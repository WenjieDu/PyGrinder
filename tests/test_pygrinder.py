"""
PyGrinder test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import numpy as np

from pygrinder import (
    mcar,
    mnar_x,
    mnar_t,
)

DEFAULT_MISSING_RATE = 0.1
NAN = 1


class TestPyGrinder(unittest.TestCase):
    complete_time_series = np.random.randn(128, 10, 36)

    def test_0_mcar(self):
        X_intact, X_with_missing, missing_mask, indicating_mask = mcar(
            self.complete_time_series, p=DEFAULT_MISSING_RATE, nan=NAN
        )
        shape_product = np.product(X_intact.shape)
        actual_missing_rate = (shape_product - missing_mask.sum()) / shape_product
        assert (
            round(actual_missing_rate, 1) == DEFAULT_MISSING_RATE
        ), f"Actual missing rate is {actual_missing_rate}, not given {DEFAULT_MISSING_RATE}"
        assert np.sum(X_with_missing[(1 - missing_mask).astype(bool)]) == NAN * np.sum(
            1 - missing_mask
        )

    def test_1_mar(self):
        pass

    def test_2_mnar(self):
        # mnar_x
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_x(
            self.complete_time_series, offset=0, nan=NAN
        )
        shape_product = np.product(X_intact.shape)
        actual_missing_rate = (shape_product - missing_mask.sum()) / shape_product
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"

        # mnar_t
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_t(
            self.complete_time_series, cycle=20, pos=10, scale=3, nan=NAN
        )
        shape_product = np.product(X_intact.shape)
        actual_missing_rate = (shape_product - missing_mask.sum()) / shape_product
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
