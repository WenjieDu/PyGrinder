# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2


def mcar_little_test(X: Union[pd.DataFrame, np.ndarray]) -> float:
    """Little's MCAR Test. Refer to :cite:`little1988TestMCAR` for more details.

    Notes
    -----
    This implementation is inspired by
    https://github.com/RianneSchouten/pyampute/blob/master/pyampute/exploration/mcar_statistical_tests.py.
    Note that this function should be used carefully. Rejecting the null hypothesis may not always mean that
    the data is not MCAR, nor is accepting the null hypothesis a guarantee that the data is MCAR.

    Parameters
    ----------
    X:
        Time series data containing missing values that should be in shape of [n_steps, n_features],
        i.e. have 2 dimensions.

    Returns
    -------
    p_value: float
        The p-value of a chi-square hypothesis test.
        Null hypothesis: the time series is missing completely at random (MCAR).

    """

    if isinstance(X, np.ndarray):
        assert len(X.shape) == 2
        dataset = pd.DataFrame(X)
    elif isinstance(X, pd.DataFrame):
        dataset = X.copy()
    else:
        raise RuntimeError(f"X should be np.ndarray or pd.DataFrame, but got {type(X)}")

    vars = dataset.dtypes.index.values
    n_features = dataset.shape[1]

    # mean and covariance estimates
    # ideally, this is done with a maximum likelihood estimator
    global_mean = dataset.mean()
    global_covariance = dataset.cov()

    # set up missing data patterns
    r = 1 * dataset.isnull()
    mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_features))))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
    dataset["mdp"] = pd.Series(correct_mdp, index=dataset.index)

    # calculate statistic and df
    pj = 0
    d2 = 0
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset["mdp"] == i, vars]
        select_vars = ~dataset_temp.isnull().any()
        pj += np.sum(select_vars)
        select_vars = vars[select_vars]
        means = dataset_temp[select_vars].mean() - global_mean[select_vars]
        select_cov = global_covariance.loc[select_vars, select_vars]
        mj = len(dataset_temp)
        parta = np.dot(
            means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1]))
        )
        d2 += mj * (np.dot(parta, means))

    df = pj - n_features

    # perform test and output p value
    p_value = 1 - chi2.cdf(d2, df)
    return p_value
