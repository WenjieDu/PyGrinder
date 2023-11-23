"""
Corrupt data by adding missing values to it with MNAR (missing not at random) pattern.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Union, Tuple, overload

import numpy as np
import torch


@overload
def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    raise NotImplementedError()


@overload
def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float,
    return_masks: bool = False,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    raise NotImplementedError()


def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float = 0,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], np.ndarray, torch.Tensor]:
    """Create not-random missing values related to values themselves (MNAR-x case ot self-masking MNAR case).
    This case follows the setting in Ipsen et al. "not-MIWAE: Deep Generative Modelling with Missing Not at Random Data"
    :cite:`ipsen2021notmiwae`.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    offset :
        the weight of standard deviation. In MNAR-x case, for each time series,
        the values larger than the mean of each time series plus offset*standard deviation will be missing

    return_masks : bool, optional, default=True
        Whether to return the masks indicating missing values in X and indicating artificially-missing values in X.
        If True, return X_intact, X, missing_mask, and indicating_mask (refer to Returns for more details).
        If False, only return X with added missing not at random values.

    nan : int/float, optional, default=0
        Value used to fill NaN values. Only valid when return_masks is True.
        If return_masks is False, the NaN values will be kept as NaN.


    Returns
    -------
    If return_masks is True:

        X_intact : array,
            Original data with missing values (nan) filled with given parameter `nan`, with observed values intact.
            X_intact is for loss calculation in the masked imputation task.

        X : array,
            Original X with artificial missing values. X is for model input.
            Both originally-missing and artificially-missing values are filled with given parameter `nan`.

        missing_mask : array,
            The mask indicates all missing values in X.
            In it, 1 indicates observed values, and 0 indicates missing values.

        indicating_mask : array,
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
        results = _mnar_x_numpy(X, offset, return_masks, nan)
    elif isinstance(X, torch.Tensor):
        results = _mnar_x_torch(X, offset, return_masks, nan)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    if not return_masks:
        X = results
        return X

    X_intact, X, missing_mask, indicating_mask = results
    return X_intact, X, missing_mask, indicating_mask


def _mnar_x_numpy(
    X: np.ndarray,
    offset: float = 0,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact

    n_s, n_l, n_c = X.shape

    ori_mask = ~np.isnan(X)
    X = np.nan_to_num(X, nan=nan)

    mask_sum = ori_mask.sum(1)
    mask_sum[mask_sum == 0] = 1
    X_mean = np.repeat(
        ((X * ori_mask).sum(1) / mask_sum).reshape(n_s, 1, n_c), n_l, axis=1
    )
    X_std = np.repeat(
        np.sqrt(np.square((X - X_mean) * ori_mask).sum(1) / mask_sum).reshape(
            n_s, 1, n_c
        ),
        n_l,
        axis=1,
    )

    mnar_missing_mask = np.zeros_like(X)
    mnar_missing_mask[X <= (X_mean + offset * X_std)] = 1
    missing_mask = ori_mask * mnar_missing_mask

    if not return_masks:
        X[missing_mask == 0] = np.nan
        return X

    indicating_mask = ori_mask - missing_mask
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X[missing_mask == 0] = nan
    return tuple((X_intact, X, missing_mask, indicating_mask))


def _mnar_x_torch(
    X: torch.Tensor,
    offset: float = 0,
    return_masks: bool = True,
    nan: Union[float, int] = 0,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    n_s, n_l, n_c = X.shape

    ori_mask = (~torch.isnan(X)).type(torch.float32)
    X = torch.nan_to_num(X, nan=nan)

    mask_sum = ori_mask.sum(1)
    mask_sum[mask_sum == 0] = 1
    X_mean = ((X * ori_mask).sum(1) / mask_sum).reshape(n_s, 1, n_c).repeat(1, n_l, 1)
    X_std = (
        (((X - X_mean) * ori_mask).pow(2).sum(1) / mask_sum)
        .sqrt()
        .reshape(n_s, 1, n_c)
        .repeat(1, n_l, 1)
    )

    mnar_missing_mask = torch.zeros_like(X)
    mnar_missing_mask[X <= (X_mean + offset * X_std)] = 1
    missing_mask = ori_mask * mnar_missing_mask

    if not return_masks:
        X[missing_mask == 0] = torch.nan
        return X

    indicating_mask = ori_mask - missing_mask
    X_intact = torch.nan_to_num(X_intact, nan=nan)
    X[missing_mask == 0] = nan
    return tuple((X_intact, X, missing_mask, indicating_mask))
