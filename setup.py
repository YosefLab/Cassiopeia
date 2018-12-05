#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


with open("README.rst") as readme_file:
    readme = readme_file.read()


requirements = [

        "numpy > 1.0, < 1.15",
        "matplotlib >= 2.2.2",
        "pandas >= 0.23.4",
        "networkx >= 2.0",
        "tqdm >= 4",
]

print(find_packages())

author = "Matthew Jones, Alex Khodaverdian, Jeffrey Quinn"

to_cythonize = [Extension("SingleCellLineageTracing/TreeSolver/lineage_solver/solver_utils", ["SingleCellLineageTracing/TreeSolver/lineage_solver/solver_utils.c"]),
                Extension("SingleCellLineageTracing/TreeSolver/simulation_tools/dataset_generation", ["SingleCellLineageTracing/TreeSolver/simulation_tools/dataset_generation.c"]),
                Extension("SingleCellLineageTracing/ProcessingPipeline/process/lineageGroup_utils", ["SingleCellLineageTracing/ProcessingPipeline/process/lineageGroup_utils.c"])]

setup(
        #ext_modules=cythonize(["TreeSolver/lineage_solver/solver_utils.pyx", "TreeSolver/simulation_tools/dataset_generation.pyx", "ProcessingPipeline/process/lineageGroup_utils.pyx"]),
        name="SingleCellLineageTracing",
        ext_modules=cythonize(to_cythonize),
        author_email="mattjones315@berkeley.edu",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
        long_description=readme + "\n\n", 
        description="Single Cell Lineage Reconstruction with Cas9-Enabled Lineage Recorders",
        install_requires=requirements,
        license="MIT license",
        include_package_data=True,
        packages=find_packages(),
        keywords="scLT",
        url="https://github.com/YosefLab/SingleCellLineageTracing",
        version='0.0.1',
        zip_safe=False,
)
