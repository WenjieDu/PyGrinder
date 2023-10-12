"""
PyGrinder package.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

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
__version__ = "0.1"

try:
    from pygrinder.mcar import mcar
    from pygrinder.utils import (
        cal_missing_rate,
        masked_fill,
    )
except Exception as e:
    print(e)


__all__ = [
    "__version__",
    "cal_missing_rate",
    "masked_fill",
    "mcar",
]
