"""
Utility functions for pygrinder.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np

try:
    import torch
except ImportError:
    pass


def cal_missing_rate(X):
    """Calculate the originally missing rate of the raw data.

    Parameters
    ----------
    X : array-like,
        Data array that may contain missing values.

    Returns
    -------
    originally_missing_rate, float,
        The originally missing rate of the raw data.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        originally_missing_rate = np.sum(np.isnan(X)) / np.product(X.shape)
    elif isinstance(X, torch.Tensor):
        originally_missing_rate = torch.sum(torch.isnan(X)) / np.product(X.shape)
        originally_missing_rate = originally_missing_rate.item()
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return originally_missing_rate


def masked_fill(X, mask, val):
    """Like torch.Tensor.masked_fill(), fill elements in given `X` with `val` where `mask` is True.

    Parameters
    ----------
    X : array-like,
        The data vector.

    mask : array-like,
        The boolean mask.

    val : float
        The value to fill in with.

    Returns
    -------
    array,
        mask
    """
    assert X.shape == mask.shape, (
        "Shapes of X and mask must match, "
        f"but X.shape={X.shape}, mask.shape={mask.shape}"
    )
    assert isinstance(X, type(mask)), (
        "Data types of X and mask must match, " f"but got {type(X)} and {type(mask)}"
    )

    if isinstance(X, list):
        X = np.asarray(X)
        mask = np.asarray(mask)

    if isinstance(X, np.ndarray):
        mask = mask.astype(bool)
        X[mask] = val
    elif isinstance(X, torch.Tensor):
        mask = mask.type(torch.bool)
        X[mask] = val
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return X
