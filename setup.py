# Copyright (C) 2022, Michael Sandborn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from setuptools import setup

setup(
    name='psig-matcher',
    version='0.1',
    packages=['psig_matcher'],
    license='GPL 3',
    description="scripts to analyze and compare piezoelectric signatures",
    long_description=open('README.md').read(),
    python_requires='>3.6',
    # do not list standard packages
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "perlin_noise",
        "plotly",
    ],
    entry_points={
        'console_scripts': [
            'psig-matcher = psig_matcher.__main__:run'
        ]
    }
)