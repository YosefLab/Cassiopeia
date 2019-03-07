#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


with open("README.md") as readme_file:
    readme = readme_file.read()


requirements = [
        "numpy > 1.0, < 1.15",
        "matplotlib >= 2.2.2",
        "pandas >= 0.22.0",
        "networkx >= 2.0",
        "tqdm >= 4",
        "gurobipy",
        'ete3 >= 3.1.1',
        'argparse >= 1.1', 
        "Biopython >= 1.71",
        "pathlib",
        'pandas-charm >= 0.1.3',
        'pysam >= 0.14.1',
        'bokeh >= 0.12.15',
        'PyYAML >= 3.12',
        'cython >= 0.29.2',
        'scipy >= 1.2.0',
        "python-Levenshtein",
        'nbconvert >= 5.4.0',
        'nbformat >= 4.4.0'
]


author = "Matthew Jones, Alex Khodaverdian, Jeffrey Quinn"

# files to wrap with cython
to_cythonize = [Extension("Cassiopeia/TreeSolver/lineage_solver/solver_utils", ["Cassiopeia/TreeSolver/lineage_solver/solver_utils.c"]),
                Extension("Cassiopeia/TreeSolver/simulation_tools/dataset_generation", ["Cassiopeia/TreeSolver/simulation_tools/dataset_generation.c"]),
                Extension("Cassiopeia/ProcessingPipeline/process/lineageGroup_utils", ["Cassiopeia/ProcessingPipeline/process/lineageGroup_utils.c"]), 
                Extension("Cassiopeia/ProcessingPipeline/process/collapse_cython", ["Cassiopeia/ProcessingPipeline/process/collapse_cython.c"]), 
                Extension("Cassiopeia/ProcessingPipeline/process/sequencing/fastq_cython", ["Cassiopeia/ProcessingPipeline/process/sequencing/fastq_cython.c"]),
                Extension("Cassiopeia/ProcessingPipeline/process/sequencing/adapters_cython", ["Cassiopeia/ProcessingPipeline/process/sequencing/adapters_cython.c"]),
                Extension("Cassiopeia/ProcessingPipeline/process/sequencing/sw_cython", ["Cassiopeia/ProcessingPipeline/process/sequencing/sw_cython.c"])]

setup(
        name="Cassiopeia",
        ext_modules=cythonize(to_cythonize),
        entry_points={
            'console_scripts': ['scLT = Cassiopeia.__main__:main',
                                'reconstruct-lineage = Cassiopeia.TreeSolver.reconstruct_tree:main',
                                'post-process-tree = Cassiopeia.TreeSolver.post_process_tree:main',
                                'stress-test = Cassiopeia.TreeSolver.stress_test:main',
                                'simulate-tree = Cassiopeia.TreeSolver.simulate_tree:main',
                                'call-lineages = Cassiopeia.ProcessingPipeline.process.lineageGroup:main',
                                'filter-molecule-table = Cassiopeia.ProcessingPipeline.process.filterMoleculeTables:main']
            
        },
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
        url="https://github.com/YosefLab/Cassiopeia",
        version='0.0.1',
        zip_safe=False,
)
