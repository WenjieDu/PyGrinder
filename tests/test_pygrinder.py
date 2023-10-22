"""
PyGrinder test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import unittest

import numpy as np

from pygrinder import (
    mcar,
    mcar_little_test,
    mar_logistic,
    mnar_x,
    mnar_t,
    masked_fill,
    cal_missing_rate,
)

DEFAULT_MISSING_RATE = 0.1
NAN = 1


class TestPyGrinder(unittest.TestCase):
    def test_0_mcar(self):
        X = np.random.randn(128, 10, 36)
        X_intact, X_with_missing, missing_mask, indicating_mask = mcar(
            X, p=DEFAULT_MISSING_RATE, nan=NAN
        )
        assert np.sum(X_with_missing[(1 - missing_mask).astype(bool)]) == NAN * np.sum(
            1 - missing_mask
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) == DEFAULT_MISSING_RATE
        ), f"Actual missing rate is {actual_missing_rate}, not given {DEFAULT_MISSING_RATE}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value: {test_pvalue}")

    def test_1_mar(self):
        X = np.random.randn(128, 36)
        X_intact, X_with_missing, missing_mask, indicating_mask = mar_logistic(
            X, obs_rate=0.1, missing_rate=0.2
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"

    def test_2_mnar(self):
        X = np.random.randn(128, 10, 36)

        # mnar_x
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_x(
            X, offset=0, nan=NAN
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"

        # mnar_t
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_t(
            X, cycle=20, pos=10, scale=3, nan=NAN
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
