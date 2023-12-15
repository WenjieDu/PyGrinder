"""
Utility functions for pygrinder.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Tuple

import numpy as np
import torch


def calc_missing_rate(X: Union[np.ndarray, torch.Tensor]) -> float:
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
        originally_missing_rate = np.sum(np.isnan(X)) / np.prod(X.shape)
    elif isinstance(X, torch.Tensor):
        originally_missing_rate = torch.sum(torch.isnan(X)) / np.prod(X.shape)
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


def fill_and_get_mask_numpy(
    X: np.ndarray,
    nan: Union[float, int] = 0,
) -> Tuple[np.ndarray, ...]:
    """Fill missing values in numpy array X with `nan` and return the missing mask.

    Parameters
    ----------
    X : np.ndarray
        Time series data generated from X_intact, with artificially missing values added.

    nan : int/float, optional, default=0
        Value used to fill NaN values. Only valid when return_masks is True.
        If return_masks is False, the NaN values will be kept as NaN.

    Returns
    -------
    X :
        Original X with artificial missing values. X is for model input.
        Both originally-missing and artificially-missing values are filled with given parameter `nan`.

    missing_mask :
        The mask indicates all missing values in X.
        In it, 1 indicates observed values, and 0 indicates missing values.

    """
    X_missing_mask = (~np.isnan(X)).astype(np.float32)
    X = np.nan_to_num(X, nan=nan)
    return X, X_missing_mask


def fill_and_get_mask_torch(
    X: torch.Tensor,
    nan: Union[float, int] = 0,
) -> Tuple[torch.Tensor, ...]:
    """Fill missing values in torch tensor X with `nan` and return the missing mask.

    Parameters
    ----------
    X :
        Time series data generated from X_intact, with artificially missing values added.

    nan : int/float, optional, default=0
        Value used to fill NaN values. Only valid when return_masks is True.
        If return_masks is False, the NaN values will be kept as NaN.

    Returns
    -------
    X :
        Original X with artificial missing values. X is for model input.
        Both originally-missing and artificially-missing values are filled with given parameter `nan`.

    missing_mask :
        The mask indicates all missing values in X.
        In it, 1 indicates observed values, and 0 indicates missing values.

    """
    missing_mask = (~torch.isnan(X)).type(torch.float32)
    X = torch.nan_to_num(X, nan=nan)
    return X, missing_mask


def fill_and_get_mask(
    X: Union[torch.Tensor, np.ndarray],
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    """Fill missing values in X with `nan` and return the missing mask.

    Parameters
    ----------
    X :
        Data with missing values

    nan : int/float, optional, default=0
        Value used to fill NaN values. Only valid when return_masks is True.
        If return_masks is False, the NaN values will be kept as NaN.

    Returns
    -------
    X :
        Original X with artificial missing values. X is for model input.
        Both originally-missing and artificially-missing values are filled with given parameter `nan`.

    missing_mask :
        The mask indicates all missing values in X.
        In it, 1 indicates observed values, and 0 indicates missing values.

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        X, missing_mask = fill_and_get_mask_numpy(X, nan)

    elif isinstance(X, torch.Tensor):
        X, missing_mask = fill_and_get_mask_torch(X, nan)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return X, missing_mask
