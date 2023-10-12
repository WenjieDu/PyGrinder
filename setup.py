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
        "missing data",
        "missing values",
        "data corruption",
        "incomplete data",
        "partial observation",
        "data mining",
        "pypots",
        "missingness",
    ],
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scikit_learn",
        "pandas",
    ],
    setup_requires=["setuptools>=38.6.0"],
)
