"""
Corrupt data by adding missing values to it with MNAR (missing not at random) pattern.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Optional, Union

import numpy as np
import torch


def _mnar_x_numpy(
    X: np.ndarray,
    offset: float = 0,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_s, n_l, n_c = X.shape
    ori_mask = ~np.isnan(X)
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
    X[missing_mask == 0] = np.nan
    return X


def _mnar_x_torch(
    X: torch.Tensor,
    offset: float = 0,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~torch.isnan(X)).type(torch.float32)
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
    X[missing_mask == 0] = torch.nan
    return X


def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
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

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mnar_x_numpy(X, offset)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mnar_x_torch(X, offset)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
