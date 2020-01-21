#!/usr/bin/env python
import os
from setuptools import setup

version = os.environ.get('VERSION', '0.0.0')

setup(
    name='ai.cge',
    description='A method for learning Causal Graph Embeddings.',
    version=version,
    author='Paul Bertin',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'ai.cge = ai.cge.main:main'
        ]
    }

)
