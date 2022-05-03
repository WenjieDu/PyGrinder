"""
Corrupt data by adding missing values to it with optional missing patterns (MCAR,MAR,MNAR).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

import numpy as np

try:
    import torch
except ImportError:
    pass


def cal_missing_rate(X):
    """ Calculate the originally missing rate of the raw data.

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
        raise TypeError('X must be type of list/numpy.ndarray/torch.Tensor, '
                        f'but got {type(X)}')

    return originally_missing_rate


def fill_nan_with_mask(X, mask):
    """ Fill missing values in ``X`` with nan according to mask.

    Parameters
    ----------
    X : array-like,
        Data vector having missing values filled with numbers (i.e. not nan).

    mask : array-like,
        Mask vector contains binary values indicating which values are missing in `data`.

    Returns
    -------
    array,
        Data vector having missing values placed with np.nan.
    """
    assert X.shape == mask.shape, 'Shapes of X and mask must match, ' \
                                  f'but X.shape={X.shape}, mask.shape={mask.shape}'
    assert type(X) == type(mask), 'Data types of X and mask must match, ' \
                                  f'but got {type(X)} and {type(mask)}'
    if isinstance(X, list):
        X = np.asarray(X)
        mask = np.asarray(mask)

    if isinstance(X, np.ndarray):
        mask = mask.astype(bool)
        X[~mask] = np.nan
    elif isinstance(X, torch.Tensor):
        mask = mask.type(torch.bool)
        X[~mask] = np.nan
    else:
        raise TypeError('X must be type of list/numpy.ndarray/torch.Tensor, '
                        f'but got {type(X)}')

    return X


def little_mcar_test(X):
    """Little's MCAR Test.

    Refer to :cite:`little1988TestMCAR`
    """
    # TODO: Little's MCAR test
    raise NotImplementedError('MCAR test has not been implemented yet.')


def mcar(X, rate, nan=0):
    """ Create completely random missing values (MCAR case).

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

    indicating_mask : array,
        The mask indicates the artificially-missing values in X, namely missing parts different from X_intact.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        return _mcar_numpy(X, rate, nan)
    elif isinstance(X, torch.Tensor):
        return _mcar_torch(X, rate, nan)
    else:
        raise TypeError('X must be type of list/numpy.ndarray/torch.Tensor, '
                        f'but got {type(X)}')


def _mcar_numpy(X, rate, nan=0):
    original_shape = X.shape
    X = X.flatten()
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
    indices = np.random.choice(indices, int(len(indices) * rate), replace=False)
    # create artificially-missing values by selected indices
    X[indices] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)
    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)
    return X_intact, X, missing_mask, indicating_mask


def _mcar_torch(X, rate, nan=0):
    original_shape = X.shape
    X = X.flatten()
    X_intact = torch.clone(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    indices = torch.where(~torch.isnan(X))[0].tolist()  # get the indices of observed values
    indices = np.random.choice(indices, int(len(indices) * rate), replace=False)
    # create artificially-missing values by selected indices
    X[indices] = torch.nan  # mask values selected by indices
    indicating_mask = ((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
    missing_mask = (~torch.isnan(X)).type(torch.float32)
    X_intact = torch.nan_to_num(X_intact, nan=nan)
    X = torch.nan_to_num(X, nan=nan)
    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)
    return X_intact, X, missing_mask, indicating_mask


def mar(X, rate, nan=0):
    """ Create random missing values (MAR case).

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

    """
    # TODO: Create missing values in MAR case
    raise NotImplementedError('MAR case has not been implemented yet.')


def mnar(X, rate, nan=0):
    """ Create not-random missing values (MNAR case).

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

    """
    # TODO: Create missing values in MNAR case
    raise NotImplementedError('MNAR case has not been implemented yet.')
