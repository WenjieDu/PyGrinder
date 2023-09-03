"""
Corrupt data by adding missing values to it with MCAR (missing completely at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np

try:
    import torch
except ImportError:
    pass


def mcar(X, rate, nan=0):
    """Create completely random missing values (MCAR case).

    Parameters
    ----------
    X : array,
        Data vector. If X has any missing values, they should be numpy.nan.

    rate : float, in (0,1),
        Artificially missing rate, rate of the observed values which will be artificially masked as missing.

        Note that,
        `rate` = (number of artificially missing values) / np.sum(~np.isnan(self.data)),
        not (number of artificially missing values) / np.product(self.data.shape),
        considering that the given data may already contain missing values,
        the latter way may be confusing because if the original missing rate >= `rate`,
        the function will do nothing, i.e. it won't play the role it has to be.

    nan : int/float, optional, default=0
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
        In it, 1 indicates artificially missing values, and other values are indicated as 0.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        return _mcar_numpy(X, rate, nan)
    elif isinstance(X, torch.Tensor):
        return _mcar_torch(X, rate, nan)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )


def _mcar_numpy(X: np.ndarray, rate: float, nan: float = 0):
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    mcar_missing_mask = np.asarray(np.random.rand(np.product(X.shape)) < rate)
    mcar_missing_mask = mcar_missing_mask.reshape(X.shape)
    X[mcar_missing_mask] = np.nan  # mask values selected by mcar_missing_mask
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)
    return X_intact, X, missing_mask, indicating_mask


def _mcar_torch(X: torch.Tensor, rate: float, nan: float = 0):
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    mcar_missing_mask = torch.rand(X.shape) < rate
    X[mcar_missing_mask] = torch.nan  # mask values selected by mcar_missing_mask
    indicating_mask = ((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
    missing_mask = (~torch.isnan(X)).type(torch.float32)
    X_intact = torch.nan_to_num(X_intact, nan=nan)
    X = torch.nan_to_num(X, nan=nan)
    return X_intact, X, missing_mask, indicating_mask


def little_mcar_test(X):
    """Little's MCAR Test.

    Refer to :cite:`little1988TestMCAR`
    """
    # TODO: Little's MCAR test
    raise NotImplementedError("MCAR test has not been implemented yet.")
