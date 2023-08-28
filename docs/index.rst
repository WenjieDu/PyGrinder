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

**A Python Toolbox for Data Corruption**

.. image:: https://img.shields.io/badge/python-v3-E97040?logo=python&logoColor=white
   :alt: Python version
.. image:: https://img.shields.io/github/v/release/wenjiedu/pycorruptor?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white
   :alt: the latest release version
   :target: https://img.shields.io/github/v/release/wenjiedu/pycorruptor?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white
.. image:: https://img.shields.io/badge/License-GPL--v3-E9BB41?logo=opensourceinitiative&logoColor=white
   :alt: License
   :target: https://github.com/WenjieDu/PyCorruptor/blob/main/LICENSE
.. image:: https://img.shields.io/github/actions/workflow/status/wenjiedu/pycorruptor/testing_ci.yml?logo=github&color=C8D8E1&label=CI
   :alt: GitHub Testing
   :target: https://github.com/WenjieDu/PyCorruptor/actions/workflows/testing_ci.yml
.. image:: https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyCorruptor?color=3C7699&label=Maintainability&logo=codeclimate
   :alt: Code Climate maintainability
   :target: https://codeclimate.com/github/WenjieDu/PyCorruptor
.. image:: https://img.shields.io/coverallsCoverage/github/WenjieDu/PyCorruptor?branch=main&logo=coveralls&color=75C1C4&label=Coverage
   :alt: Coveralls report
   :target: https://coveralls.io/github/WenjieDu/PyCorruptor
.. image:: https://img.shields.io/conda/dn/conda-forge/pycorruptor?label=Conda%20Downloads&color=AED0ED&logo=anaconda&logoColor=white
   :alt: Conda downloads
   :target: https://anaconda.org/conda-forge/pypots
.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FWenjieDu%2FWenjieDu%2Fmain%2Ffigs%2Fprojects%2Fpycorruptor_downloads.json
   :alt: PyPI downloads
   :target: https://pepy.tech/project/pycorruptor
.. image:: https://img.shields.io/badge/Contributor%20Covenant-v2.1-4baaaa.svg
   :alt: CODE of CONDUCT
.. image:: https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FPyCorruptor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits+since+May+2022&edge_flat=false
   :alt: Visit num

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