"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import torch


def random_select_start_indices(
    features,
    p,
    n_samples,
    n_steps,
    n_features,
):
    if features is None:
        all_feature_indices = list(range(n_samples * n_features))
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]
    else:
        all_feature_indices = [
            i * n_features + j for i in range(n_samples) for j in features
        ]
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]

    selected_feature_start_indices = np.random.choice(
        all_feature_start_indices,
        round(len(all_feature_start_indices) * p),
        replace=False,
    )
    return selected_feature_start_indices


def _seq_missing_numpy(
    X: np.ndarray,
    p: float,
    seq_len: int,
    features: list = None,
    steps: list = None,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_samples, n_steps, n_features = X.shape
    start_indices = random_select_start_indices(
        features, p, n_samples, n_steps, n_features
    )

    X = X.transpose(0, 2, 1)
    X = X.reshape(-1)
    for idx in start_indices:
        idx += np.random.choice(
            steps,
            1,
            replace=False,
        )[0]
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(0, 2, 1)
    return X


def _seq_missing_torch(
    X: torch.Tensor,
    p: float,
    seq_len: int,
    features: list = None,
    steps: list = None,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_samples, n_steps, n_features = X.shape
    start_indices = random_select_start_indices(
        features, p, n_samples, n_steps, n_features
    )

    X = X.transpose(1, 2)
    X = X.flatten()
    for idx in start_indices:
        idx += np.random.choice(
            steps,
            1,
            replace=False,
        )[0]
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(1, 2)
    return X


def seq_missing(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    seq_len: int,
    features: list = None,
    steps: list = None,
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(X, list):
        X = np.asarray(X)
    n_samples, n_steps, n_features = X.shape

    assert 0 < p <= 1, f"p must be in range (0, 1), but got {p}"
    assert isinstance(
        seq_len, int
    ), f"`seq_len` must be type of int, but got {type(seq_len)}"
    if features is not None:
        assert isinstance(
            features, list
        ), f"`features` must be type of list, but got {type(features)}"

        assert (
            max(features) <= n_features
        ), f"values in `features` must be <= {n_features}, but got {max(features)}"

    if steps is not None:
        assert isinstance(
            steps, list
        ), f"`steps` must be type of list, but got {type(steps)}"

        assert (
            max(steps) <= n_steps
        ), f"values in `steps` must be <= {n_steps}, but got {max(steps)}"
        assert (
            n_steps - max(steps) >= seq_len
        ), f"n_steps - max(steps) must be >= seq_len, but got {n_steps - max(steps)}"
    else:
        steps = list(range(n_steps - seq_len))

    if isinstance(X, np.ndarray):
        corrupted_X = _seq_missing_numpy(
            X,
            p,
            seq_len,
            features,
            steps,
        )
    elif isinstance(X, torch.Tensor):
        corrupted_X = _seq_missing_torch(
            X,
            p,
            seq_len,
            features,
            steps,
        )
    else:
        raise TypeError(
            "X must be type of list/numpy.ndarray/torch.Tensor, " f"but got {type(X)}"
        )

    return corrupted_X
