"""
Corrupt data by adding missing values to it with MCAR (missing completely at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Tuple, overload

import numpy as np
import torch


@overload
def mcar(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    raise NotImplementedError()


@overload
def mcar(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    return_masks: bool = False,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    raise NotImplementedError()


def mcar(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    """Create completely random missing values (MCAR case).

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p : float, in (0,1),
        The probability that values may be masked as missing completely at random.
        Note that the values are randomly selected no matter if they are originally missing or observed.
        If the selected values are originally missing, they will be kept as missing.
        If the selected values are originally observed, they will be masked as missing.
        Therefore, if the given X already contains missing data, the final missing rate in the output X could be
        in range [original_missing_rate, original_missing_rate+rate], but not strictly equal to
        `original_missing_rate+rate`. Because the selected values to be artificially masked out may be originally
        missing, and the masking operation on the values will do nothing.

    return_masks : bool, optional, default=True
        Whether to return the masks indicating missing values in X and indicating artificially-missing values in X.
        If True, return X_intact, X, missing_mask, and indicating_mask (refer to Returns for more details).
        If False, only return X with added missing at completely random values.

    nan : int/float, optional, default=0
        Value used to fill NaN values. Only valid when return_masks is True.
        If return_masks is False, the NaN values will be kept as NaN.

    Returns
    -------
    If return_masks is True:

        X_intact : array-like
            Original data with missing values (nan) filled with given parameter `nan`, with observed values intact.
            X_intact is for loss calculation in the masked imputation task.

        X : array-like
            Original X with artificial missing values.
            Both originally-missing and artificially-missing values are filled with given parameter `nan`.

        missing_mask : array-like
            The mask indicates all missing values in X.
            In it, 1 indicates observed values, and 0 indicates missing values.

        indicating_mask : array-like
            The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
            In it, 1 indicates artificially missing values,
            and the other values (including originally observed/missing values) are indicated as 0.

    If return_masks is False:

        X : array-like
            Original X with artificial missing values.
            Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        results = _mcar_numpy(X, p, return_masks, nan)
    elif isinstance(X, torch.Tensor):
        results = _mcar_torch(X, p, return_masks, nan)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    if not return_masks:  # only return X with MCAR values if not return masks
        X = results
        return X

    X_intact, X, missing_mask, indicating_mask = results
    return X_intact, X, missing_mask, indicating_mask


def _mcar_numpy(
    X: np.ndarray,
    p: float,
    return_masks: bool,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    mcar_missing_mask = np.asarray(np.random.rand(np.product(X.shape)) < p)
    mcar_missing_mask = mcar_missing_mask.reshape(X.shape)
    X[mcar_missing_mask] = np.nan  # mask values selected by mcar_missing_mask

    if not return_masks:  # return X with MCAR values only if not return masks
        return X

    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)
    return tuple((X_intact, X, missing_mask, indicating_mask))


def _mcar_torch(
    X: torch.Tensor,
    p: float,
    return_masks: bool,
    nan: Union[float, int] = 0,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    mcar_missing_mask = torch.rand(X.shape) < p
    X[mcar_missing_mask] = torch.nan  # mask values selected by mcar_missing_mask

    if not return_masks:  # return X with MCAR values only if not return masks
        return X

    indicating_mask = ((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
    missing_mask = (~torch.isnan(X)).type(torch.float32)
    X_intact = torch.nan_to_num(X_intact, nan=nan)
    X = torch.nan_to_num(X, nan=nan)
    return tuple((X_intact, X, missing_mask, indicating_mask))
