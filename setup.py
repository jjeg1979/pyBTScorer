"""Minimal setup file for pybtscorer project."""

from setuptools import setup, find_packages

setup(
    name='pybtscorer',
    version='0.1.0',
    license='proprietary',
    description='Minimal Project for Backtesting Scoring',

    author='J.J. Expósito',
    author_email='',
    url='',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=[],
    extras_require={},

    entry_points={
        'console_scripts': [
            
        ]
    },
)