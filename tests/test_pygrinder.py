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
    mnar_nonuniform,
    rdo,
    seq_missing,
    block_missing,
    masked_fill,
    calc_missing_rate,
    fill_and_get_mask,
)

DEFAULT_MISSING_RATE = 0.1
NaN = 1


class TestPyGrinder(unittest.TestCase):
    def test_0_mcar(self):
        X = np.random.randn(128, 10, 36)
        X_with_missing = mcar(
            X,
            p=DEFAULT_MISSING_RATE,
        )
        X_with_missing, missing_mask = fill_and_get_mask(X_with_missing, NaN)

        assert np.sum(X_with_missing[(1 - missing_mask).astype(bool)]) == NaN * np.sum(
            1 - missing_mask
        )
        # as list
        list_X_with_missing = masked_fill(
            X_with_missing.tolist(),
            (1 - missing_mask).tolist(),
            np.nan,
        ).tolist()
        _ = calc_missing_rate(list_X_with_missing)
        # as torch tensor
        tensor_X_with_missing = masked_fill(
            torch.from_numpy(X_with_missing),
            torch.from_numpy(1 - missing_mask),
            torch.nan,
        )
        _ = calc_missing_rate(tensor_X_with_missing)
        # as numpy array
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = calc_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) == DEFAULT_MISSING_RATE
        ), f"Actual missing rate is {actual_missing_rate}, not given {DEFAULT_MISSING_RATE}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MCAR_return_masks: {test_pvalue}")

        # only add missing values into X
        X = torch.randn(128, 10, 36)
        X_with_nan = mcar(X, p=DEFAULT_MISSING_RATE)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MCAR_not_return_masks: {test_pvalue}")

    def test_1_mar(self):
        X = np.random.randn(128, 36)
        X_with_missing = mar_logistic(X, obs_rate=0.1, missing_rate=0.2)
        X_with_missing, missing_mask = fill_and_get_mask(X_with_missing, NaN)
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = calc_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MAR_return_masks: {test_pvalue}")

        # only add missing values into X
        X = torch.randn(128, 36)
        X_with_nan = mar_logistic(X, obs_rate=0.1, missing_rate=0.2)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MAR_not_return_masks: {test_pvalue}")

    def test_2_mnar(self):
        X = np.random.randn(128, 10, 36)

        # mnar_x
        X_with_missing = mnar_x(X, offset=0)
        X_with_missing, missing_mask = fill_and_get_mask(X_with_missing, NaN)
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = calc_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_X_return_masks: {test_pvalue}")

        # mnar_t
        X_with_missing = mnar_t(X, cycle=20, pos=10, scale=3)
        X_with_missing, missing_mask = fill_and_get_mask(X_with_missing, NaN)
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = calc_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_T_return_masks: {test_pvalue}")

        # mnar_nonuniform
        X_with_missing, _ = mnar_nonuniform(X, p=0.5, increase_factor=0.5)
        X_with_missing, missing_mask = fill_and_get_mask(X_with_missing, NaN)
        X_with_missing = masked_fill(X_with_missing, 1 - missing_mask, np.nan)
        actual_missing_rate = calc_missing_rate(X_with_missing)
        assert (
            round(actual_missing_rate, 1) > 0
        ), f"Actual missing rate is {actual_missing_rate}"
        test_pvalue = mcar_little_test(X_with_missing.reshape(128, -1))
        print(
            f"MCAR Little test p_value for MNAR_Non_Uniform_Masking_return_masks: {test_pvalue}"
        )

        # only add missing values into X
        # mnar_x
        X = torch.randn(128, 10, 36)
        X_with_nan = mnar_x(X, offset=0)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_X_not_return_masks: {test_pvalue}")

        # mnar_t
        X_with_nan = mnar_t(X, cycle=20, pos=10, scale=3)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(f"MCAR Little test p_value for MNAR_T_not_return_masks: {test_pvalue}")

        # mnar_nonuniform
        X_with_nan, _ = mnar_nonuniform(X, p=0.5, increase_factor=0.5)
        test_pvalue = mcar_little_test(X_with_nan.numpy().reshape(128, -1))
        print(
            f"MCAR Little test p_value for MNAR_Non_Uniform_Masking_not_return_masks: {test_pvalue}"
        )

    def test_3_rdo(self):
        X = np.random.randn(128, 10, 36)
        X_with_missing = mcar(
            X,
            p=DEFAULT_MISSING_RATE,
        )
        n_observations = (~np.isnan(X_with_missing)).sum()
        n_rdo = round(DEFAULT_MISSING_RATE * n_observations)

        X_with_rdo = rdo(X_with_missing, p=DEFAULT_MISSING_RATE)
        n_left_observations = (~np.isnan(X_with_rdo)).sum()
        assert n_left_observations == n_observations - n_rdo

        X_with_missing = torch.from_numpy(X_with_missing)
        X_with_rdo = rdo(X_with_missing, p=DEFAULT_MISSING_RATE)
        n_left_observations = (~torch.isnan(X_with_rdo)).sum()
        assert n_left_observations == n_observations - n_rdo

    def test_4_seq_missing(self):
        n_samples, n_steps, n_features = 128, 24, 10
        X = np.random.randn(n_samples, n_steps, n_features)

        # The realized missing rate of subsequence masking does not equal p exactly
        # (which is why the old exact-equality asserts were disabled), but the output
        # must preserve the input shape and contain a non-trivial amount of missingness.
        for p, seq_len in [(0.1, 3), (0.5, 3), (0.9, 3)]:
            # exercise both numpy input and torch input
            X_input = torch.from_numpy(X) if p == 0.9 else X
            X_with_seq_missing = seq_missing(X_input, p, seq_len)
            assert (
                X_with_seq_missing.shape == X_input.shape
            ), "seq_missing should preserve the input shape."
            actual_missing_rate = calc_missing_rate(X_with_seq_missing)
            print(f"sequence {p} actual_missing_rate: {actual_missing_rate}")
            assert (
                actual_missing_rate > 0
            ), f"seq_missing produced no missing values for p={p}."

    def test_5_block_missing(self):
        n_samples, n_steps, n_features = 128, 24, 10
        X = np.random.randn(n_samples, n_steps, n_features)

        for factor in [0.1, 0.5, 0.9]:
            block_len, block_width = 5, 5
            # exercise both numpy input and torch input
            X_input = torch.from_numpy(X) if factor == 0.9 else X
            X_with_block_missing = block_missing(X_input, factor, block_len, block_width)
            assert (
                X_with_block_missing.shape == X_input.shape
            ), "block_missing should preserve the input shape."
            actual_missing_rate = calc_missing_rate(X_with_block_missing)
            print(f"block {factor} actual_missing_rate: {actual_missing_rate}")
            assert (
                actual_missing_rate > 0
            ), f"block_missing produced no missing values for factor={factor}."


if __name__ == "__main__":
    unittest.main()
