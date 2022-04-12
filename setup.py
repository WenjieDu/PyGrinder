from setuptools import setup, find_packages

from pycorruptor.__version__ import version

with open('./README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='pycorruptor',
    version=version,
    description='A Python Toolbox for Data Corruption',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Wenjie Du',
    author_email='wenjay.du@gmail.com',
    url='https://github.com/WenjieDu/PyCorruptor',
    download_url='https://github.com/WenjieDu/PyCorruptor/archive/master.zip',
    keywords=[
        'missing data', 'missing values', 'data corruption',
        'incomplete data', 'partially observed',
        'data mining',
    ],
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scikit_learn',
        'pandas',
    ],
    setup_requires=['setuptools>=38.6.0'],
)
