.. PyCorruptor documentation master file, created by
sphinx-quickstart on Wed Mar 15 18:14:27 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to PyCorruptor's documentation!
=======================================
.. image:: https://raw.githubusercontent.com/WenjieDu/PyCorruptor/main/docs/figs/PyCorruptor.svg?sanitize=true
   :height: 300
   :align: right
   :target: https://github.com/WenjieDu/PyCorruptor
   :alt: PyCorruptor logo

.. centered:: A Python Toolbox for Data Corruption

.. image:: https://img.shields.io/badge/python-v3-yellowgreen
   :alt: Python version
.. image:: https://img.shields.io/pypi/v/pycorruptor?color=green&label=PyPI
   :alt: PyPI version
   :target: https://pypi.org/project/pycorruptor
.. image:: https://static.pepy.tech/personalized-badge/pycorruptor?period=total&units=none&left_color=gray&right_color=blue&left_text=Total%20Downloads
   :alt: PyPI download number
.. image:: https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FPyCorruptor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits&edge_flat=false
   :alt: Visit number

In data analysis and modeling, sometimes we may need to corrupt the original data to achieve our goal, for instance, evaluating models' ability to reconstruct corrupted data or assessing the model's performance on only partially-observed data. PyCorruptor is such a tool to help you corrupt your data, which provides several patterns to create missing values in the given data.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Code Documentation

   pycorruptor

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   faq
   about_us


References
""""""""""
.. bibliography::