"""
Corrupt data by adding missing values to it with MNAR (missing not at random) pattern.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from typing import Optional, Union, Tuple

import numpy as np

try:
    import torch
except ImportError:
    pass


def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float = 0,
    nan: Union[float, int] = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Create not-random missing values related to values themselves (MNAR-x case ot self-masking MNAR case).
    This case follows the setting in Ipsen et al. "not-MIWAE: Deep Generative Modelling with Missing Not at Random Data"
    :cite`ipsen2021notmiwae`.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    offset :
        the weight of standard deviation. In MNAR-x case, for each time series,
        the values larger than the mean of each time series plus offset*standard deviation will be missing

    nan :
        Value used to fill NaN values.

    Returns
    -------
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

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        X_intact, X, missing_mask, indicating_mask = _mnar_x_numpy(X, offset, nan)
    elif isinstance(X, torch.Tensor):
        X_intact, X, missing_mask, indicating_mask = _mnar_x_torch(X, offset, nan)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )
    return X_intact, X, missing_mask, indicating_mask


def _mnar_x_numpy(
    X: np.ndarray,
    offset: float = 0,
    nan: Union[float, int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact

    n_s, n_l, n_c = X.shape

    ori_mask = ~np.isnan(X)
    X = np.nan_to_num(X, nan=nan)
    X_intact = np.nan_to_num(X_intact, nan=nan)

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
    indicating_mask = ori_mask - missing_mask
    X[missing_mask == 0] = nan
    return X_intact, X, missing_mask, indicating_mask


def _mnar_x_torch(
    X: torch.Tensor,
    offset: float = 0,
    nan: Union[float, int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    n_s, n_l, n_c = X.shape

    ori_mask = (~torch.isnan(X)).type(torch.float32)
    X = torch.nan_to_num(X, nan=nan)
    X_intact = torch.nan_to_num(X_intact, nan=nan)

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
    indicating_mask = ori_mask - missing_mask
    X[missing_mask == 0] = nan
    return X_intact, X, missing_mask, indicating_mask


def mnar_t(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
    nan: Union[float, int] = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Create not-random missing values related to temporal dynamics (MNAR-t case).
    In particular, the missingness is generated by an intensity function f(t) = exp(3*torch.sin(cycle*t + pos)).
    This case mainly follows the setting in https://hawkeslib.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    cycle :
        The cycle of the used intensity function

    pos :
        The displacement of the used intensity function

    scale :
        The scale number to control the missing rate

    nan :
        Value used to fill NaN values.

    Returns
    -------
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

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        X_intact, X, missing_mask, indicating_mask = _mnar_t_numpy(
            X, cycle, pos, scale, nan
        )
    elif isinstance(X, torch.Tensor):
        X_intact, X, missing_mask, indicating_mask = _mnar_t_torch(
            X, cycle, pos, scale, nan
        )
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )
    return X_intact, X, missing_mask, indicating_mask


def _mnar_t_numpy(
    X: np.ndarray,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
    nan: Union[float, int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact

    n_s, n_l, n_c = X.shape

    ori_mask = (~np.isnan(X)).astype(np.float32)
    X = np.nan_to_num(X, nan=nan)
    X_intact = np.nan_to_num(X_intact, nan=nan)

    ts = np.linspace(0, 1, n_l).reshape(1, n_l, 1)
    ts = np.repeat(ts, n_s, axis=0)
    ts = np.repeat(ts, n_c, axis=2)
    intensity = np.exp(3 * np.sin(cycle * ts + pos))
    mnar_missing_mask = (np.random.rand(n_s, n_l, n_c) * scale) < intensity

    missing_mask = ori_mask * mnar_missing_mask
    indicating_mask = ori_mask - missing_mask
    X[missing_mask == 0] = nan
    return X_intact, X, missing_mask, indicating_mask


def _mnar_t_torch(
    X: torch.Tensor,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
    nan: Union[float, int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    n_s, n_l, n_c = X.shape

    ori_mask = (~torch.isnan(X)).type(torch.float32)
    X = torch.nan_to_num(X, nan=nan)
    X_intact = torch.nan_to_num(X_intact, nan=nan)

    ts = torch.linspace(0, 1, n_l).reshape(1, n_l, 1).repeat(n_s, 1, n_c)
    intensity = torch.exp(3 * torch.sin(cycle * ts + pos))
    mnar_missing_mask = (torch.randn(X.size()).uniform_(0, 1) * scale) < intensity

    missing_mask = ori_mask * mnar_missing_mask
    indicating_mask = ori_mask - missing_mask
    X[missing_mask == 0] = nan
    return X_intact, X, missing_mask, indicating_mask
