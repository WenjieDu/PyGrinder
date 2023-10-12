"""
Corrupt data by adding missing values to it with MAR (missing at random) pattern.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


def mar(X, rate, nan=0):
    """Create random missing values (MAR case).

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
    raise NotImplementedError("MAR case has not been implemented yet.")


def _mar_numpy(X, rate, nan=0):
    pass


def _mar_torch(X, rate, nan=0):
    pass
