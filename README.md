<a href='https://github.com/WenjieDu/PyCorruptor'><img src='https://raw.githubusercontent.com/WenjieDu/PyCorruptor/main/docs/figs/PyCorruptor.svg?sanitize=true' width='350' align='right' /></a>

# <p align='center'>Welcome to PyCorruptor</p>
**<p align='center'>A Python Toolbox for Data Corruption</p>**
<p align='center'>
    <!-- Python version -->
    <img src='https://img.shields.io/badge/python-v3-yellowgreen'>
    <!-- PyPI version -->
    <img alt="PyPI" src="https://img.shields.io/pypi/v/pycorruptor?color=green&label=PyPI">
    <!-- GitHub Testing -->
    <a alt='GitHub Testing' href='https://github.com/WenjieDu/PyCorruptor/actions/workflows/testing.yml'> 
        <img src='https://github.com/WenjieDu/PyCorruptor/actions/workflows/testing.yml/badge.svg'>
    </a>
    <!-- Coveralls report -->
    <a alt='Coveralls report' href='https://coveralls.io/github/WenjieDu/PyCorruptor'> 
        <img src='https://img.shields.io/coverallsCoverage/github/WenjieDu/PyCorruptor?branch=main&logo=coveralls'>
    </a>
    <a href="https://anaconda.org/conda-forge/pycorruptor">
        <img alt="Conda downloads" src="https://img.shields.io/conda/dn/conda-forge/pycorruptor?label=Conda%20Downloads&color=AED0ED&logo=anaconda&logoColor=white">
    </a>
    <a href="https://pypi.org/project/pycorruptor">
        <img alt="PyPI downloads" src="https://static.pepy.tech/personalized-badge/pycorruptor?period=total&units=international_system&left_color=grey&right_color=teal&left_text=PyPI%20Downloads&logo=github">
    </a>
    <!-- Visit number -->
    
</p>

<a href='https://github.com/WenjieDu/PyPOTS'><img src='https://raw.githubusercontent.com/WenjieDu/PyPOTS/main/docs/_static/figs/PyPOTS_logo.svg?sanitize=true' width='160' align='left' /></a>
PyCorruptor is a part of [PyPOTS project](https://github.com/WenjieDu/PyPOTS) (a Python toolbox for data mining on Partially-Observed Time Series), and was separated from PyPOTS for decoupling missingness-creating functionalities from learning algorithms.


In data analysis and modeling, sometimes we may need to corrupt the original data to achieve our goal, for instance, evaluating models' ability to reconstruct corrupted data or assessing the model's performance on only partially-observed data. PyCorruptor is such a tool to help you corrupt your data, which provides several patterns to create missing values in the given data.

## ❖ Citing PyCorruptor
The paper introducing PyPOTS project is available on arXiv at [this URL](https://arxiv.org/abs/2305.18811),
and we are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). If you use PyCorruptor in your work,
please cite PyPOTS project as below and 🌟star this repository to make others notice this library. 🤗 Thank you!

``` bibtex
@article{du2023PyPOTS,
title={{PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series}},
author={Wenjie Du},
year={2023},
eprint={2305.18811},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2305.18811},
doi={10.48550/arXiv.2305.18811},
}
```

or

> Wenjie Du. (2023).
> PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811. https://doi.org/10.48550/arXiv.2305.18811

<details>
<summary>🏠 Visits</summary>
<img align='left' src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FPyCorruptor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits&edge_flat=false'>
</details>