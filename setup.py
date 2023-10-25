from setuptools import setup, find_packages

from pygrinder import __version__

with open("./README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="pygrinder",
    version=__version__,
    description="A Python toolkit for introducing missing values into datasets",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    author="Wenjie Du",
    author_email="wenjay.du@gmail.com",
    url="https://github.com/WenjieDu/PyGrinder",
    download_url="https://github.com/WenjieDu/PyGrinder/archive/main.zip",
    keywords=[
        "data corruption",
        "incomplete data",
        "data mining",
        "pypots",
        "missingness",
        "partially observed",
        "irregular sampled",
        "partially-observed time series",
        "incomplete time series",
        "missing data",
        "missing values",
        "pypots",
    ],
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
    ],
    setup_requires=["setuptools>=38.6.0"],
)
