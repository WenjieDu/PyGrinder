"""
Corrupt data by adding missing values to it with MCAR (missing completely at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch


def _mcar_numpy(
    X: np.ndarray,
    p: float,
) -> np.ndarray:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    mcar_missing_mask = np.asarray(np.random.rand(np.prod(X.shape)) < p)
    mcar_missing_mask = mcar_missing_mask.reshape(X.shape)
    X[mcar_missing_mask] = np.nan  # mask values selected by mcar_missing_mask
    return X


def _mcar_torch(
    X: torch.Tensor,
    p: float,
) -> torch.Tensor:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    mcar_missing_mask = torch.rand(X.shape) < p
    X[mcar_missing_mask] = torch.nan  # mask values selected by mcar_missing_mask
    return X


def mcar(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create completely random missing values (MCAR case).

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p :
        The probability that values may be masked as missing completely at random.
        Note that the values are randomly selected no matter if they are originally missing or observed.
        If the selected values are originally missing, they will be kept as missing.
        If the selected values are originally observed, they will be masked as missing.
        Therefore, if the given X already contains missing data, the final missing rate in the output X could be
        in range [original_missing_rate, original_missing_rate+rate], but not strictly equal to
        `original_missing_rate+rate`. Because the selected values to be artificially masked out may be originally
        missing, and the masking operation on the values will do nothing.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mcar_numpy(X, p)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mcar_torch(X, p)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
