from setuptools import setup, find_packages

from pycorruptor import __version__

with open("./README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="pycorruptor",
    version=__version__,
    description="A Python Toolbox for Data Corruption",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GPL-3.0",
    author="Wenjie Du",
    author_email="wenjay.du@gmail.com",
    url="https://github.com/WenjieDu/PyCorruptor",
    download_url="https://github.com/WenjieDu/PyCorruptor/archive/main.zip",
    keywords=[
        "missing data",
        "missing values",
        "data corruption",
        "incomplete data",
        "partial observation",
        "data mining",
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
