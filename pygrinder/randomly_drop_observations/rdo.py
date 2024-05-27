"""
Corrupt data by randomly drop original observations.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch


def _rdo_numpy(
    X: np.ndarray,
    p: float,
) -> np.ndarray:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    ori_shape = X.shape
    X = X.reshape(-1)
    indices = np.where(~np.isnan(X))[0].tolist()
    indices = np.random.choice(
        indices,
        round(len(indices) * p),
        replace=False,
    )
    X[indices] = np.nan
    X = X.reshape(ori_shape)
    return X


def _rdo_torch(
    X: torch.Tensor,
    p: float,
) -> torch.Tensor:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    ori_shape = X.shape
    X = X.reshape(-1)
    indices = torch.where(~torch.isnan(X))[0].tolist()
    indices = np.random.choice(
        indices,
        round(len(indices) * p),
        replace=False,
    )
    X[indices] = torch.nan
    X = X.reshape(ori_shape)
    return X


def rdo(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create missingness in the data by randomly drop observations.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p :
        The proportion of the observed values that will be randomly masked as missing.
        RDO (randomly drop observations) will randomly select values from the observed values to be masked as missing.
        The number of selected observations is determined by `p` and the total number of observed values in X,
        e.g. if `p`=0.1, and there are 1000 observed values in X, then 0.1*1000=100 values will be randomly selected
        to be masked as missing. If the result is not an integer, the number of selected values will be rounded to
        the nearest.

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
        corrupted_X = _rdo_numpy(X, p)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _rdo_torch(X, p)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
