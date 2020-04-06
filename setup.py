#!/usr/bin/env python
import os
from setuptools import setup

"""
The first time you need to run:
 pip install -e .
"""

version = os.environ.get('VERSION', '0.0.0')

setup(
    name='ai.causalcell',
    description='A method for learning Causal Graph Embeddings.',
    version=version,
    author='Paul Bertin',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ai.causalcell = ai.causalcell.main:main'
        ]
    }, install_requires=['click', 'skopt', 'tqdm', 'numpy', 'torch', 'PyYAML', 'pandas', 'rdkit', 'cmapPy']

)
