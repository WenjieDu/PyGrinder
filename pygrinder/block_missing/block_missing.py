"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import math
from typing import Union

import numpy as np
import torch


def random_select_start_indices(
    block_width,
    feature_idx,
    step_idx,
    p,
    n_samples,
    n_steps,
    n_features,
) -> np.ndarray:
    all_feature_indices = [
        i * n_features + j for i in range(n_samples) for j in feature_idx
    ]
    all_feature_start_indices = [i * n_steps for i in all_feature_indices]
    selected_feature_start_indices = np.random.choice(
        all_feature_start_indices,
        math.ceil(len(all_feature_start_indices) * p),
        replace=False,
    )
    selected_feature_start_indices = np.asarray(selected_feature_start_indices)

    step_shift = np.random.choice(
        step_idx,
        len(selected_feature_start_indices),
    )
    step_shift = np.asarray(step_shift)

    selected_start_indices = selected_feature_start_indices + step_shift
    selected_start_indices = [
        i + j * n_steps for i in selected_start_indices for j in range(block_width)
    ]
    return np.asarray(selected_start_indices)


def _block_missing_numpy(
    X: np.ndarray,
    p: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_samples, n_steps, n_features = X.shape
    start_indices = random_select_start_indices(
        block_width, feature_idx, step_idx, p, n_samples, n_steps, n_features
    )

    X = X.transpose(0, 2, 1)
    X = X.reshape(-1)
    for idx in start_indices:
        X[idx : idx + block_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(0, 2, 1)
    return X


def _block_missing_torch(
    X: torch.Tensor,
    p: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_samples, n_steps, n_features = X.shape
    start_indices = random_select_start_indices(
        block_width, feature_idx, step_idx, p, n_samples, n_steps, n_features
    )

    X = X.transpose(1, 2)
    X = X.flatten()
    for idx in start_indices:
        X[idx : idx + block_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(1, 2)
    return X


def block_missing(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(X, list):
        X = np.asarray(X)
    n_samples, n_steps, n_features = X.shape

    assert 0 < p <= 1, f"p must be in range (0, 1), but got {p}"

    assert isinstance(
        block_len, int
    ), f"`block_len` must be type of int, but got {type(block_len)}"
    assert block_len <= n_steps, f"`seq_len` must be <= {n_steps}, but got {block_len}"

    assert isinstance(
        block_width, int
    ), f"`block_width` must be type of int, but got {type(block_width)}"
    assert (
        block_width <= n_features
    ), f"`block_width` must be <= {n_features}, but got {block_width}"

    if feature_idx is not None:
        assert isinstance(
            feature_idx, list
        ), f"`feature_idx` must be type of list, but got {type(feature_idx)}"

        assert (
            max(feature_idx) <= n_features
        ), f"values in `feature_idx` must be <= {n_features}, but got {max(feature_idx)}"
    else:
        feature_idx = list(range(n_features - block_width))

    if step_idx is not None:
        assert isinstance(
            step_idx, list
        ), f"`step_idx` must be type of list, but got {type(step_idx)}"

        assert (
            max(step_idx) <= n_steps
        ), f"values in `step_idx` must be <= {n_steps}, but got {max(step_idx)}"
        assert (
            n_steps - max(step_idx) >= block_len
        ), f"n_steps - max(step_idx) must be >= block_len, but got {n_steps - max(step_idx)}"
    else:
        step_idx = list(range(n_steps - block_len))

    if isinstance(X, np.ndarray):
        corrupted_X = _block_missing_numpy(
            X,
            p,
            block_len,
            block_width,
            feature_idx,
            step_idx,
        )
    elif isinstance(X, torch.Tensor):
        corrupted_X = _block_missing_torch(
            X,
            p,
            block_len,
            block_width,
            feature_idx,
            step_idx,
        )
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X
