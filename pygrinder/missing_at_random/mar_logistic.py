"""
Corrupt data by adding missing values to it with MAR (missing at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch
from scipy import optimize


def _mar_logistic_torch(
    X: Union[np.ndarray, torch.Tensor],
    rate_obs: float,
    rate_missing: float,
) -> Union[np.ndarray, torch.Tensor]:
    def pick_coefficients(X, idxs_obs=None, idxs_nas=None, self_mask=False):
        n, d = X.shape
        if self_mask:
            coeffs = torch.randn(d)
            Wx = X * coeffs
            coeffs /= torch.std(Wx, 0)
        else:
            d_obs = len(idxs_obs)
            d_na = len(idxs_nas)
            coeffs = torch.randn(d_obs, d_na)
            Wx = X[:, idxs_obs].mm(coeffs)
            coeffs /= torch.std(Wx, 0, keepdim=True)
        return coeffs

    def fit_intercepts(X, coeffs, p, self_mask=False):
        if self_mask:
            d = len(coeffs)
            intercepts = torch.zeros(d)
            for j in range(d):

                def f(x):
                    return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            d_obs, d_na = coeffs.shape
            intercepts = torch.zeros(d_na)
            for j in range(d_na):

                def f(x):
                    return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        return intercepts

    assert len(X.shape) == 2, "X should be 2 dimensional"
    n, d = X.shape

    ori_type_is_np = isinstance(X, np.ndarray)
    if ori_type_is_np:
        X = torch.from_numpy(X).to(torch.float32)
    else:
        X = torch.clone(X).to(torch.float32)

    assert (
        torch.isnan(X).sum() == 0
    ), "the input X of the mar_logistic() shouldn't containing originally missing data"

    mask = torch.zeros(n, d).bool()

    # number of variables that will have no missing values (at least one variable)
    d_obs = max(int(rate_obs * d), 1)
    d_na = d - d_obs  # number of variables that will have missing values

    # Sample variables will all be observed, and the left will be with missing values
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coefficients(X, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, rate_missing)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)
    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps  # True means missing

    X[mask] = torch.nan

    return X.numpy() if ori_type_is_np else X


def mar_logistic(
    X: Union[torch.Tensor, np.ndarray],
    obs_rate: float,
    missing_rate: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create random missing values (MAR case) with a logistic model.
    First, a subset of the variables without missing values is randomly selected.
    Missing values will be introduced into the remaining variables according to a logistic model with random weights.
    This implementation is inspired by the tutorial
    https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values

    Parameters
    ----------
    X :
        A time series data vector without any missing data. Shape of [n_steps, n_features].

    obs_rate :
        The proportion of variables without missing values that will be used for fitting the logistic masking model.

    missing_rate:
        The proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
        corrupted_X = _mar_logistic_torch(X, missing_rate, obs_rate)
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
