"""
Corrupt data by adding missing values to it with MNAR (missing not at random) pattern.
"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch


def _mnar_t_numpy(
    X: np.ndarray,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~np.isnan(X)).astype(np.float32)
    ts = np.linspace(0, 1, n_l).reshape(1, n_l, 1)
    ts = np.repeat(ts, n_s, axis=0)
    ts = np.repeat(ts, n_c, axis=2)
    intensity = np.exp(3 * np.sin(cycle * ts + pos))
    mnar_missing_mask = (np.random.rand(n_s, n_l, n_c) * scale) < intensity
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = np.nan
    return X


def _mnar_t_torch(
    X: torch.Tensor,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~torch.isnan(X)).type(torch.float32)
    ts = torch.linspace(0, 1, n_l).reshape(1, n_l, 1).repeat(n_s, 1, n_c)
    intensity = torch.exp(3 * torch.sin(cycle * ts + pos))
    mnar_missing_mask = (torch.randn(X.size()).uniform_(0, 1) * scale) < intensity
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = torch.nan
    return X


def mnar_t(
    X: Union[np.ndarray, torch.Tensor],
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> Union[np.ndarray, torch.Tensor]:
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


    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mnar_t_numpy(X, cycle, pos, scale)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mnar_t_torch(X, cycle, pos, scale)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )
    return corrupted_X
