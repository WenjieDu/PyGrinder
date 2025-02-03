"""
Corrupt data by adding missing values to it with MNAR (missing not at random) pattern based on non uniform masking.
"""

# Created by Linglong Qian <linglong.qian@kcl.ac.uk>
# License: BSD-3-Clause

from typing import Union, Tuple, Optional

import numpy as np
import torch


def _adjust_probability_vectorized(
    obs_count: Union[int, float],
    avg_count: Union[int, float],
    base_prob: float,
    increase_factor: float = 0.5,
) -> float:
    """Adjusts the base probability based on observed and average counts using a scaling factor.

    Parameters
    ----------
    obs_count :
        The observed count of an event or observation in the dataset.
    avg_count :
        The average count of the event or observation across the dataset.
    base_prob :
        The base probability of the event or observation occurring.
    increase_factor :
        A scaling factor applied to adjust the probability when `obs_count` is below `avg_count`.
        This factor influences how much to increase or decrease the probability.

    Returns
    -------
    float :
        The adjusted probability, scaled based on the ratio between the observed count and the average count.
        The adjusted probability will be within the range [0, 1].
    """
    if obs_count < avg_count:
        # Increase probability when observed count is lower than average count
        return min(base_prob * (avg_count / obs_count) * increase_factor, 1.0)
    else:
        # Decrease probability when observed count exceeds average count
        return max(base_prob * (obs_count / avg_count) / increase_factor, 0.0)


def _mnar_nonuniform_numpy(
    X: np.ndarray,
    p: float,
    pre_replacement_probabilities: Optional[np.ndarray] = None,
    increase_factor: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create MNAR missing values based on numerical features using NumPy.

    Parameters
    ----------
    X :
        Data array of shape [N, T, D].
    p :
        The probability of masking values as missing.
    pre_replacement_probabilities :
        Pre-defined replacement probabilities for each feature.
    increase_factor :
        Factor to adjust replacement probabilities based on observation counts.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        Tuple containing corrupted array and replacement probabilities used.
    """
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    N, T, D = X.shape

    # Compute replacement probabilities if not provided
    if pre_replacement_probabilities is None:
        observations_per_feature = np.sum(~np.isnan(X), axis=(0, 1))
        average_observations = np.mean(observations_per_feature)
        replacement_probabilities = np.full(D, p)

        if increase_factor > 0:
            for feature_idx in range(D):
                replacement_probabilities[feature_idx] = _adjust_probability_vectorized(
                    observations_per_feature[feature_idx],
                    average_observations,
                    replacement_probabilities[feature_idx],
                    increase_factor=increase_factor,
                )

            total_observations = np.sum(observations_per_feature)
            total_replacement_target = total_observations * p

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = np.sum(
                    replacement_probabilities * observations_per_feature
                )
                if np.isclose(total_replacement, total_replacement_target, rtol=1e-3):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
    else:
        replacement_probabilities = pre_replacement_probabilities

    # Randomly remove data points based on replacement probabilities
    random_matrix = np.random.rand(N, T, D)

    # masking all values(except original nan) with probability
    # X[(~np.isnan(X)) & (random_matrix < replacement_probabilities)] = np.nan

    # masking all values(including original nan) with probability
    X[random_matrix < replacement_probabilities] = np.nan

    return X, replacement_probabilities


def _mnar_nonuniform_torch(
    X: torch.Tensor,
    p: float,
    pre_replacement_probabilities: Optional[torch.Tensor] = None,
    increase_factor: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create MNAR missing values based on numerical features using PyTorch.

    Parameters
    ----------
    X :
        Data tensor of shape [N, T, D].
    p :
        The probability of masking values as missing.
    pre_replacement_probabilities :
        Pre-defined replacement probabilities for each feature.
    increase_factor :
        Factor to adjust replacement probabilities based on observation counts.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor] :
        Tuple containing corrupted tensor and replacement probabilities used.
    """
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    N, T, D = X.shape

    # Compute replacement probabilities if not provided
    if pre_replacement_probabilities is None:
        observations_per_feature = torch.sum(~torch.isnan(X), dim=(0, 1))
        average_observations = torch.mean(observations_per_feature.float())
        replacement_probabilities = torch.full((D,), p)

        if increase_factor > 0:
            for feature_idx in range(D):
                replacement_probabilities[feature_idx] = _adjust_probability_vectorized(
                    observations_per_feature[feature_idx].item(),
                    average_observations.item(),
                    replacement_probabilities[feature_idx].item(),
                    increase_factor=increase_factor,
                )

            total_observations = torch.sum(observations_per_feature)
            total_replacement_target = total_observations * p

            for _ in range(1000):  # Limit iterations to prevent infinite loop
                total_replacement = torch.sum(
                    replacement_probabilities * observations_per_feature
                )
                if torch.isclose(
                    total_replacement, total_replacement_target, rtol=1e-3
                ):
                    break
                adjustment_factor = total_replacement_target / total_replacement
                replacement_probabilities *= adjustment_factor
    else:
        replacement_probabilities = pre_replacement_probabilities

    # Randomly remove data points based on replacement probabilities
    random_matrix = torch.rand(N, T, D)

    # masking all values(except original nan) with probability
    # X[(~torch.isnan(X)) & (random_matrix < replacement_probabilities)] = torch.nan

    # masking all values(including original nan) with probability
    X[random_matrix < replacement_probabilities] = torch.nan

    return X, replacement_probabilities


def mnar_nonuniform(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    pre_replacement_probabilities: Optional[Union[np.ndarray, torch.Tensor]] = None,
    increase_factor: float = 0.5,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Create not-random missing values based on numerical features (MNAR-non-uniform case).
    Missing values are introduced based on the observation counts of features, with adjustable
    probabilities that can be increased for under-observed features.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.
    p :
        The probability that values may be masked as missing. Must be between 0 and 1.
        Note that this is the target probability - actual probabilities for each feature
        will be adjusted based on their observation counts.
    pre_replacement_probabilities :
        Pre-defined replacement probabilities for each feature. If provided, these probabilities
        will be used instead of computing new ones.
    increase_factor :
        Factor to adjust replacement probabilities based on observation counts. Higher values
        will increase the probability of removing values from under-observed features.

    Returns
    -------
    Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]] :
        Tuple containing:
        - corrupted_X: Original X with artificial missing values
        - replacement_probabilities: The probabilities used for removing values
    """
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        if pre_replacement_probabilities is not None and isinstance(
            pre_replacement_probabilities, torch.Tensor
        ):
            pre_replacement_probabilities = pre_replacement_probabilities.numpy()
        corrupted_X, probs = _mnar_nonuniform_numpy(
            X, p, pre_replacement_probabilities, increase_factor
        )
    elif isinstance(X, torch.Tensor):
        if pre_replacement_probabilities is not None and isinstance(
            pre_replacement_probabilities, np.ndarray
        ):
            pre_replacement_probabilities = torch.from_numpy(
                pre_replacement_probabilities
            )
        corrupted_X, probs = _mnar_nonuniform_torch(
            X, p, pre_replacement_probabilities, increase_factor
        )
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X, probs
