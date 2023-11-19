<a href='https://github.com/WenjieDu/PyGrinder'><img src='https://pypots.com/figs/pypots_logos/PyGrinder_logo_FFBG.svg?sanitize=true' width='200' align='right' /></a>

<h2 align="center">Welcome to PyGrinder</h2>

*<p align='center'>a Python toolkit for grinding data beans into the incomplete</p>*

<p align='center'>
    <a href='https://github.com/WenjieDu/PyGrinder'>
        <img alt='Python version' src='https://img.shields.io/badge/python-v3-E97040?logo=python&logoColor=white'>
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/PyGrinder?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS/blob/main/README.md#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pygrinder?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://star-history.com/#wenjiedu/pygrinder">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pygrinder?logo=Github&color=6BB392&label=Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pygrinder?logo=Github&color=91B821&label=Forks">
    </a>
    <a href="https://codeclimate.com/github/WenjieDu/PyGrinder">
        <img alt="Code Climate maintainability" src="https://img.shields.io/codeclimate/maintainability-percentage/WenjieDu/PyGrinder?color=3C7699&label=Maintainability&logo=codeclimate">
    </a>
    <a href='https://coveralls.io/github/WenjieDu/PyGrinder'>
        <img alt='Coveralls report' src='https://img.shields.io/coverallsCoverage/github/WenjieDu/PyGrinder?branch=main&logo=coveralls&color=75C1C4&label=Coverage'>
    </a>
    <a  href='https://github.com/WenjieDu/PyGrinder/actions/workflows/testing_ci.yml'>
        <img alt='GitHub Testing' src='https://img.shields.io/github/actions/workflow/status/wenjiedu/PyGrinder/testing_ci.yml?logo=github&color=C8D8E1&label=CI'>
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
    <a href="https://anaconda.org/conda-forge/PyGrinder">
        <img alt="Conda downloads" src="https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/conda_pygrinder_downloads.json">
    </a>
    <a href='https://pepy.tech/project/PyGrinder'>
        <img alt='PyPI downloads' src='https://img.shields.io/endpoint?url=https://pypots.com/figs/downloads_badges/pypi_pygrinder_downloads.json'>
    </a>
</p>

<a href='https://github.com/WenjieDu/PyPOTS'><img src='https://pypots.com/figs/pypots_logos/PyPOTS_logo_FFBG.svg?sanitize=true' width='160' align='left' /></a>
PyGrinder is a part of 
<a href="https://github.com/WenjieDu/PyPOTS">
PyPOTS <img align="center" src="https://img.shields.io/github/stars/WenjieDu/PyPOTS?style=social">
</a>
(a Python toolbox for data mining on
Partially-Observed Time Series), was called PyCorruptor and separated from PyPOTS for decoupling missingness-creating functionalities from
learning algorithms.

In data analysis and modeling, sometimes we may need to corrupt the original data to achieve our goal, for instance,
evaluating models' ability to reconstruct corrupted data or assessing the model's performance on only partially-observed
data. PyGrinder is such a tool to help you corrupt your data, which provides several patterns to create missing values
in the given data.


## ❖ Usage Examples
PyGrinder now is available on <a alt='Anaconda' href='https://anaconda.org/conda-forge/tsdb'><img align='center' src='https://img.shields.io/badge/Anaconda--lightgreen?style=social&logo=anaconda'></a>❗️

Install it with `conda install pygrinder`, you may need to specify the channel with option `-c conda-forge`

or install from PyPI:
> pip install pygrinder

or install from source code:
> pip install `https://github.com/WenjieDu/PyGrinder/archive/main.zip`

```python
import numpy as np
import pygrinder

# given a time-series dataset with 128 samples, each sample with 10 time steps and 36 data features
ts_dataset = np.random.randn(128, 10, 36)

# grind the dataset with MCAR pattern, 10% missing probability, and using 0 to fill missing values
X_intact, X, missing_mask, indicating_mask = pygrinder.mcar(ts_dataset, p=0.1, nan=0)

# grind the dataset with MAR pattern
X_intact, X, missing_mask, indicating_mask = pygrinder.mar_logistic(ts_dataset[:, 0, :], obs_rate=0.1, missing_rate=0.1, nan=0)

# grind the dataset with MNAR pattern
X_intact, X, missing_mask, indicating_mask = pygrinder.mnar_x(ts_dataset, offset=0.1, nan=0)
X_intact, X, missing_mask, indicating_mask = pygrinder.mnar_t(ts_dataset, cycle=20, pos = 10, scale = 3, nan=0)
```


## ❖ Citing PyGrinder/PyPOTS

The paper introducing PyPOTS project is available on arXiv at [this URL](https://arxiv.org/abs/2305.18811),
and we are pursuing to publish it in prestigious academic venues, e.g. JMLR (track for
[Machine Learning Open Source Software](https://www.jmlr.org/mloss/)). If you use PyGrinder in your work,
please cite PyPOTS project as below and 🌟star this repository to make others notice this library. 🤗 Thank you!


``` bibtex
@article{du2023PyPOTS,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
author={Wenjie Du},
year={2023},
eprint={2305.18811},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2305.18811},
doi={10.48550/arXiv.2305.18811},
}
```

> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> arXiv, abs/2305.18811.https://arxiv.org/abs/2305.18811

or

``` bibtex
@inproceedings{du2023PyPOTS,
title={{PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series}},
booktitle={9th SIGKDD workshop on Mining and Learning from Time Series (MiLeTS'23)},
author={Wenjie Du},
year={2023},
url={https://arxiv.org/abs/2305.18811},
}
```

> Wenjie Du. (2023).
> PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series.
> In *9th SIGKDD workshop on Mining and Learning from Time Series (MiLeTS'23)*. https://arxiv.org/abs/2305.18811


<details>
<summary>🏠 Visits</summary>
<img align='left' src='https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWenjieDu%2FPyCorruptor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visits+since+May+2022&edge_flat=false'>
</details>
