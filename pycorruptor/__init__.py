"""
PyCorruptor package
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from .__version__ import version as __version__

from .corrupt import (
    cal_missing_rate,
    fill_nan_with_mask,
    mcar,
)
