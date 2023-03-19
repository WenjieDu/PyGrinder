"""
PyCorruptor package
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

from pycorruptor.__version__ import version as __version__

try:
    from pycorruptor.corrupt import (
        cal_missing_rate,
        masked_fill,
        mcar,
    )

except Exception as e:
    print(e)
