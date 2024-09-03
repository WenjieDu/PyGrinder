"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Union

import numpy as np
import torch
from tsdb.utils.logging import logger


def random_select_start_indices(
    feature_idx,
    step_idx,
    hit_rate,
    n_samples,
    n_steps,
    n_features,
) -> np.ndarray:
    if feature_idx is None:
        all_feature_indices = list(range(n_samples * n_features))
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]
    else:
        all_feature_indices = [
            i * n_features + j for i in range(n_samples) for j in feature_idx
        ]
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]

    if hit_rate > 1:
        logger.warning(f"hit_rate={hit_rate} > 1")

    selected_feature_start_indices = np.random.choice(
        all_feature_start_indices,
        math.ceil(len(all_feature_start_indices) * hit_rate),
        replace=hit_rate > 1,
    )
    selected_feature_start_indices = np.asarray(selected_feature_start_indices)

    step_shift = np.random.choice(
        step_idx,
        len(selected_feature_start_indices),
    )
    step_shift = np.asarray(step_shift)

    selected_start_indices = selected_feature_start_indices + step_shift
    return selected_start_indices


def _seq_missing_numpy(
    X: np.ndarray,
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = p * n_steps / seq_len
    start_indices = random_select_start_indices(
        feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(0, 2, 1)
    X = X.reshape(-1)
    for idx in start_indices:
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(0, 2, 1)
    return X


def _seq_missing_torch(
    X: torch.Tensor,
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = p * n_steps / seq_len
    start_indices = random_select_start_indices(
        feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(1, 2)
    X = X.flatten()
    for idx in start_indices:
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(1, 2)
    return X


def seq_missing(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Create subsequence missing data.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p :
        The probability that values may be masked as missing completely at random.

    seq_len :
        The length of missing sequence.

    feature_idx :
        The indices of features for missing sequences to be corrupted.

    step_idx :
        The indices of steps for a missing sequence to start with.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)
    n_samples, n_steps, n_features = X.shape

    assert 0 < p <= 1, f"p must be in range (0, 1), but got {p}"
    assert isinstance(
        seq_len, int
    ), f"`seq_len` must be type of int, but got {type(seq_len)}"
    assert seq_len <= n_steps, f"`seq_len` must be <= {n_steps}, but got {seq_len}"

    if feature_idx is not None:
        assert isinstance(
            feature_idx, list
        ), f"`feature_idx` must be type of list, but got {type(feature_idx)}"

        assert (
            max(feature_idx) <= n_features
        ), f"values in `feature_idx` must be <= {n_features}, but got {max(feature_idx)}"

    if step_idx is not None:
        assert isinstance(
            step_idx, list
        ), f"`step_idx` must be type of list, but got {type(step_idx)}"

        assert (
            max(step_idx) <= n_steps
        ), f"values in `step_idx` must be <= {n_steps}, but got {max(step_idx)}"
        assert (
            n_steps - max(step_idx) >= seq_len
        ), f"n_steps - max(step_idx) must be >= seq_len, but got {n_steps - max(step_idx)}"
    else:
        step_idx = list(range(n_steps - seq_len + 1))

    if isinstance(X, np.ndarray):
        corrupted_X = _seq_missing_numpy(
            X,
            p,
            seq_len,
            feature_idx,
            step_idx,
        )
    elif isinstance(X, torch.Tensor):
        corrupted_X = _seq_missing_torch(
            X,
            p,
            seq_len,
            feature_idx,
            step_idx,
        )
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X
