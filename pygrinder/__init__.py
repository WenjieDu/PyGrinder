"""
PyGrinder: a Python toolkit for grinding data beans into the incomplete.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# PyGrinder version
#
# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
__version__ = "0.5"

from .missing_at_random import mar_logistic
from .missing_completely_at_random import mcar, mcar_little_test
from .missing_not_at_random import mnar_x, mnar_t
from .randomly_drop_observations import rdo
from .utils import (
    calc_missing_rate,
    masked_fill,
    fill_and_get_mask,
    fill_and_get_mask_torch,
    fill_and_get_mask_numpy,
)

__all__ = [
    "__version__",
    "mcar",
    "mcar_little_test",
    "mar_logistic",
    "mnar_x",
    "mnar_t",
    "rdo",
    "calc_missing_rate",
    "masked_fill",
    "fill_and_get_mask",
    "fill_and_get_mask_torch",
    "fill_and_get_mask_numpy",
]
