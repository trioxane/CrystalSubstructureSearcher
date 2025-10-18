#!/usr/bin/env python
# coding: utf-8

"""
Setup configuration for CrystalSubstructureSearch package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    # ===== PACKAGE INFORMATION =====
    name='CrystalSubstructureSearcher',
    version='1.0.0',
    description='Tool for analyzing crystal structures and identifying low-periodic substructures',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # ===== AUTHOR INFORMATION =====
    author='Pavel N. Zolotarev',
    author_email='pavel.zolotarev@unimib.it',
    
    # ===== PROJECT METADATA =====
    url='https://github.com/trioxane/CrystalSubstructureSearcher',
    project_urls={
        'Documentation': 'https://github.com/trioxane/CrystalSubstructureSearcher#readme',
        'Source Code': 'https://github.com/trioxane/CrystalSubstructureSearcher',
    },
    license='MIT',
    
    # ===== PYTHON VERSION =====
    python_requires='>=3.9',
    
    # ===== PACKAGE LOCATION =====
    # find_packages() finds all folders with __init__.py
    # Since css/ is in root, it will be found automatically
    packages=find_packages(),
    
    # ===== PACKAGE DATA =====
    # Include CSV and other data files in the package
    package_data={
        'css': [
            'data/*.csv',  # Include all CSV files from css/data/
        ],
    },
    include_package_data=True,
    
    # ===== DEPENDENCIES =====
    # These will be installed automatically when users install your package
    install_requires=[
        'mendeleev==0.14.0',
        'networkx==2.8.4',
        'numpy==1.25.0',
        'pandas==2.0.3',
        'pymatgen==2025.1.24',
    ],

    # ===== CLASSIFIERS =====
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    
    # ===== KEYWORDS =====
    keywords='crystal structure substructure analysis materials science',
)
