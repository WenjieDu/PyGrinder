"""
Utility functions for pygrinder.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch


def cal_missing_rate(X: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate the originally missing rate of the raw data.

    Parameters
    ----------
    X:
        Data array that may contain missing values.

    Returns
    -------
    originally_missing_rate,
        The originally missing rate of the raw data. Its value should be in the range [0,1].

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


def masked_fill(
    X: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    val: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Like torch.Tensor.masked_fill(), fill elements in given `X` with `val` where `mask` is True.

    Parameters
    ----------
    X:
        The data vector.

    mask:
        The boolean mask.

    val:
        The value to fill in with.

    Returns
    -------
    filled_X:
        Mask filled X.

    """
    if isinstance(X, list):
        X = np.asarray(X)
        mask = np.asarray(mask)

    assert X.shape == mask.shape, (
        "Shapes of X and mask must match, "
        f"but X.shape={X.shape}, mask.shape={mask.shape}"
    )
    assert isinstance(X, type(mask)), (
        "Data types of X and mask must match, " f"but got {type(X)} and {type(mask)}"
    )

    if isinstance(X, np.ndarray):
        filled_X = X.copy()
        mask = mask.copy()
        mask = mask.astype(bool)
        filled_X[mask] = val
    elif isinstance(X, torch.Tensor):
        filled_X = torch.clone(X)
        mask = torch.clone(mask)
        mask = mask.type(torch.bool)
        filled_X[mask] = val
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return filled_X
