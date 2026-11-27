# PyGrinder <a href="https://github.com/WenjieDu/PyGrinder"><img src="https://pypots.com/figs/pypots_logos/PyGrinder/logo_FFBG.svg" width="200" align="right" alt="PyGrinder logo"></a>

> A Python Toolkit for Simulating Missing Values in Real-World Data

<p>
    <a href="https://docs.pypots.com/en/latest/pygrinder.html">
        <img alt="Python version" src="https://img.shields.io/badge/Python-v3.10+-F8C6B5?logo=python&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/releases">
        <img alt="the latest release version" src="https://img.shields.io/github/v/release/wenjiedu/pygrinder?color=EE781F&include_prereleases&label=Release&logo=github&logoColor=white">
    </a>
    <a href="https://docs.pypots.com/en/latest/pygrinder.html">
        <img alt="Docs building" src="https://img.shields.io/readthedocs/pypots?logo=readthedocs&label=Docs&logoColor=white&color=395260">
    </a>
    <a href="https://pepy.tech/project/pygrinder">
        <img alt="PyPI downloads" src="https://pypots.com/figs/downloads_badges/pypi_pygrinder_downloads.svg">
    </a>
    <a href="https://anaconda.org/conda-forge/pygrinder">
        <img alt="Conda downloads" src="https://pypots.com/figs/downloads_badges/conda_pygrinder_downloads.svg">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/blob/main/LICENSE">
        <img alt="BSD-3 license" src="https://img.shields.io/badge/License-BSD--3-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://github.com/WenjieDu/PyPOTS#-community">
        <img alt="Community" src="https://img.shields.io/badge/join_us-community!-C8A062">
    </a>
    <a href="https://star-history.com/#wenjiedu/pygrinder">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/wenjiedu/pygrinder?logo=None&color=6BB392&label=%E2%98%85%20Stars">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/wenjiedu/pygrinder?logo=forgejo&logoColor=black&label=Forks">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/graphs/contributors">
        <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/wenjiedu/pygrinder?color=D8E699&label=Contributors&logo=GitHub">
    </a>
    <a href="https://github.com/WenjieDu/PyGrinder/actions/workflows/testing_ci.yml">
        <img alt="GitHub Testing" src="https://img.shields.io/github/actions/workflow/status/wenjiedu/pygrinder/testing_ci.yml?logo=circleci&color=C8D8E1&label=CI">
    </a>
    <a href="https://coveralls.io/github/WenjieDu/PyGrinder?branch=main">
        <img alt="Coveralls coverage" src="https://img.shields.io/coverallsCoverage/github/WenjieDu/PyGrinder?branch=main&logo=coveralls&color=75C1C4&label=Coverage">
    </a>
    <a href="https://sonarcloud.io/component_measures?id=WenjieDu_PyGrinder&metric=sqale_rating&view=list">
        <img alt="maintainability" src="https://sonarcloud.io/api/project_badges/measure?project=WenjieDu_PyGrinder&metric=sqale_rating">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Code Style" src="https://img.shields.io/badge/Code_Style-Ruff-D7FF64?logo=ruff">
    </a>
    <a href="https://arxiv.org/abs/2305.18811">
        <img alt="arXiv DOI" src="https://img.shields.io/badge/DOI-10.48550/arXiv.2305.18811-F8F7F0">
    </a>
</p>

PyGrinder is an open-source Python toolkit for **simulating missing values and manipulating partially observed data**.
It is the missingness-simulation layer of the [PyPOTS ecosystem](https://pypots.com/ecosystem/), separated from
PyPOTS so data corruption and benchmarking utilities can be used independently of model implementations.

PyGrinder implements common statistical missingness mechanisms—including **MCAR**, **MAR**, and several **MNAR**
patterns—together with time-structured corruption strategies such as randomly dropped observations, sequential
missingness, and block missingness. Core corruption functions support both NumPy arrays and PyTorch tensors.

[Website](https://pypots.com) · [Documentation](https://docs.pypots.com/en/latest/pygrinder.html) · [Tutorials](https://github.com/WenjieDu/BrewPOTS) · [Community](https://github.com/WenjieDu/PyPOTS#-community)

[**Key Features**](#-key-features) · [**Installation**](#-installation) · [**Usage**](#-usage) · [**Missingness Patterns**](#-missingness-patterns) · [**PyPOTS Ecosystem**](#-pypots-ecosystem) · [**Citing PyGrinder/PyPOTS**](#-citing-pygrinderpypots) · [**Contribution**](#-contribution)

## ❖ Key Features

- **Classical missingness mechanisms:** Simulate missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR).
- **Time-structured corruption:** Create randomly dropped observations, sequential missing segments, and block-wise missingness for time-series experiments.
- **NumPy and PyTorch support:** Core corruption APIs accept NumPy arrays or PyTorch tensors and represent missing values with `NaN`.
- **Missingness utilities:** Calculate missing rates, apply masks, and fill values while retaining observation masks for downstream learning.
- **Lightweight and reusable:** Use PyGrinder independently for simulation studies or together with BenchPOTS and PyPOTS for reproducible time-series experiments.
- **Research-oriented APIs:** Simple functional interfaces make missingness generation easy to reproduce, inspect, and combine with custom evaluation pipelines.

## ❖ Installation

PyGrinder requires Python 3.10 or later and is available on [PyPI](https://pypi.org/project/pygrinder/)
and [Anaconda](https://anaconda.org/conda-forge/pygrinder).

Choose **one** of the following installation methods.

**Option 1: pip (recommended)**

```bash
pip install pygrinder
# To upgrade: pip install pygrinder --upgrade
```

**Option 2: Conda**

```bash
conda install conda-forge::pygrinder
# To upgrade: conda update conda-forge::pygrinder
```

**Option 3: Latest source code on main (may include unreleased changes)**

```bash
pip install https://github.com/WenjieDu/PyGrinder/archive/main.zip
```

## ❖ Usage

### Quick Start

The example below generates a complete synthetic time series and introduces 10% MCAR missingness.

```python
import numpy as np

from pygrinder import calc_missing_rate, mcar

rng = np.random.default_rng(42)
X = rng.normal(size=(128, 24, 8)).astype(np.float32)

X_missing = mcar(X, p=0.1)

print(X_missing.shape)              # (128, 24, 8)
print(calc_missing_rate(X_missing)) # approximately 0.10
```

PyGrinder also works directly with PyTorch tensors:

```python
import torch

from pygrinder import seq_missing

X = torch.randn(32, 24, 8)
X_missing = seq_missing(X, p=0.1, seq_len=4)
```

### Mask and Fill Utilities

For workflows that need both filled data and the original observation mask:

```python
import numpy as np

from pygrinder import fill_and_get_mask

X = np.array([[1.0, np.nan], [3.0, 4.0]])
X_filled, missing_mask = fill_and_get_mask(X, 0.0)
```

See the [PyGrinder API documentation](https://docs.pypots.com/en/latest/pygrinder.html) for function signatures and parameter details.

## ❖ Missingness Patterns

| Pattern | API | Description |
|:--|:--|:--|
| MCAR | `mcar` | Randomly mask observations independently of the data values |
| MCAR diagnostic | `mcar_little_test` | Perform Little's test for evaluating an MCAR assumption |
| MAR | `mar_logistic` | Generate missingness through a logistic MAR mechanism |
| MNAR-X | `mnar_x` | Generate value-dependent MNAR missingness |
| MNAR-T | `mnar_t` | Generate time-dependent MNAR missingness |
| Non-uniform MNAR | `mnar_nonuniform` | Apply feature-dependent non-uniform replacement probabilities |
| Randomly dropped observations | `rdo` | Randomly remove a proportion of observed values |
| Sequential missingness | `seq_missing` | Mask contiguous sequences along the time dimension |
| Block missingness | `block_missing` | Mask contiguous time-feature blocks |

Utility APIs include `calc_missing_rate`, `masked_fill`, `fill_and_get_mask`,
`fill_and_get_mask_numpy`, and `fill_and_get_mask_torch`.

## ❖ PyPOTS Ecosystem

Welcome to the PyPOTS coffee universe! Each library plays a distinct role in the time-series analysis pipeline.

- [**TSDB**](https://github.com/WenjieDu/TSDB) makes public time-series datasets easy to discover, download, and load.
- **PyGrinder** simulates missingness and creates partially observed data for controlled experiments.
- [**BenchPOTS**](https://github.com/WenjieDu/BenchPOTS) provides standardized preprocessing pipelines for reproducible benchmarking.
- [**PyPOTS**](https://github.com/WenjieDu/PyPOTS) provides unified modeling for imputation, forecasting, classification, clustering, anomaly detection, and representation learning.
- [**BrewPOTS**](https://github.com/WenjieDu/BrewPOTS) provides tutorials and notebooks for learning the ecosystem.

<p align="center">
<a href="https://pypots.com/ecosystem/">
    <img src="https://pypots.com/figs/pypots_logos/Ecosystem/PyPOTS_Ecosystem_Pipeline.png" width="95%" alt="PyPOTS ecosystem pipeline"/>
</a>
<br>
<b> ☕️ Welcome to the universe of PyPOTS. Enjoy it and have fun!</b>
</p>

## ❖ Citing PyGrinder/PyPOTS

PyGrinder is developed as part of the PyPOTS ecosystem. If it supports your research, please cite the
[PyPOTS toolkit paper](https://arxiv.org/abs/2305.18811v2).

```bibtex
@article{du2023pypots,
title = {{PyPOTS: A Python Toolkit for Data Mining on Partially-Observed Time Series}},
author = {Wenjie Du},
journal = {SIGKDD MiLeTS Workshop},
url = {https://arxiv.org/abs/2305.18811v1},
year = {2023},
}
```

```bibtex
@article{du2025pypots,
title = {{PyPOTS: A Python Toolkit for Machine Learning on Partially-Observed Time Series}},
author = {Wenjie Du and Yiyuan Yang and Linglong Qian and Jun Wang and Qingsong Wen},
journal = {arXiv preprint arXiv:2305.18811v2},
url = {https://arxiv.org/abs/2305.18811v2},
year = {2025},
}
```

## ❖ Contribution

We welcome bug fixes, new missingness mechanisms, documentation improvements, and regression tests.
Report reproducible problems or propose improvements through [GitHub Issues](https://github.com/WenjieDu/PyGrinder/issues).

Before submitting a pull request, enable [pre-commit](https://pre-commit.com/) in your local clone:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

GitHub CI checks Ruff linting and formatting, package construction, dependency changes, and the supported Python matrix.
If you find PyGrinder useful, consider starring the repository to help others discover it.

## ❖ Community

Join the [PyPOTS community](https://github.com/WenjieDu/PyPOTS#-community) for questions, research discussion,
feedback, and ecosystem development. You can also open a [PyGrinder issue](https://github.com/WenjieDu/PyGrinder/issues)
for reproducible bugs or feature requests.
