"""
PyGrinder test cases.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import unittest

import numpy as np
import torch

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
NaN = 1


class TestPyGrinder(unittest.TestCase):
    def test_0_mcar(self):
        X = np.random.randn(128, 10, 36)
        X_intact, X_with_missing, missing_mask, indicating_mask = mcar(
            X,
            p=DEFAULT_MISSING_RATE,
            return_masks=True,
            nan=NaN,
        )
        assert np.sum(X_with_missing[(1 - missing_mask).astype(bool)]) == NaN * np.sum(
            1 - missing_mask
        )
        # as list
        list_X_with_missing = masked_fill(
            X_with_missing.tolist(),
            (1 - missing_mask).tolist(),
            np.nan,
        ).tolist()
        _ = cal_missing_rate(list_X_with_missing)
        # as torch tensor
        tensor_X_with_missing = masked_fill(
            torch.from_numpy(X_with_missing),
            torch.from_numpy(1 - missing_mask),
            torch.nan,
        )
        _ = cal_missing_rate(tensor_X_with_missing)
        # as numpy array
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) == DEFAULT_MISSING_RATE
        ), f"Actual missing rate is {actual_missing_rate}, not given {DEFAULT_MISSING_RATE}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MCAR_return_masks: {test_pvalue}")

        # only add missing values into X
        X = torch.randn(128, 10, 36)
        X_with_nan = mcar(X, p=DEFAULT_MISSING_RATE, return_masks=False)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MCAR_not_return_masks: {test_pvalue}")

    def test_1_mar(self):
        X = np.random.randn(128, 36)
        X_intact, X_with_missing, missing_mask, indicating_mask = mar_logistic(
            X, obs_rate=0.1, missing_rate=0.2, return_masks=True, nan=NaN
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MAR_return_masks: {test_pvalue}")

        # only add missing values into X
        X = torch.randn(128, 36)
        X_with_nan = mar_logistic(X, obs_rate=0.1, missing_rate=0.2, return_masks=False)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MAR_not_return_masks: {test_pvalue}")

    def test_2_mnar(self):
        X = np.random.randn(128, 10, 36)

        # mnar_x
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_x(
            X, offset=0, return_masks=True, nan=NaN
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_X_return_masks: {test_pvalue}")

        # mnar_t
        X_intact, X_with_missing, missing_mask, indicating_mask = mnar_t(
            X, cycle=20, pos=10, scale=3, return_masks=True, nan=NaN
        )
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = cal_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_T_return_masks: {test_pvalue}")

        # only add missing values into X
        # mnar_x
        X = torch.randn(128, 10, 36)
        X_with_nan = mnar_x(X, offset=0, return_masks=False)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_X_not_return_masks: {test_pvalue}")
        # mnar_t
        X_with_nan = mnar_t(X, cycle=20, pos=10, scale=3, return_masks=False)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_T_not_return_masks: {test_pvalue}")
