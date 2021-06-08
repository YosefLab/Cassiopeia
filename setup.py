#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, find_packages
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext


with open("README.md") as readme_file:
    readme = readme_file.read()


requirements = [
    "numba >= 0.51.0"
    "numpy > 1.17",
    "matplotlib >= 2.2.2",
    "pandas >= 0.22.0",
    "networkx >= 2.5",
    "tqdm >= 4",
    # "gurobipy",
    "ete3 >= 3.1.1",
    "argparse >= 1.1",
    "Biopython >= 1.71",
    "pandas-charm >= 0.1.3",
    "pysam >= 0.14.1",
    "bokeh >= 0.12.15",
    "PyYAML >= 3.12",
    "cython >= 0.29.2",
    "scipy >= 1.2.0",
    "python-Levenshtein",
    "nbconvert >= 5.4.0",
    "nbformat >= 4.4.0",
    "hits",
    "scikit-bio >= 0.5.6",
]


author = "Matthew Jones, Alex Khodaverdian, Richard Zhang, Sebastian Prillo"

cmdclass = {"build_ext": build_ext}

# files to wrap with cython
to_cythonize = [
    Extension(
        "cassiopeia.preprocess.doublet_utils",
        ["cassiopeia/preprocess/doublet_utils.pyx"],
    ),
    Extension(
        "cassiopeia.preprocess.map_utils",
        ["cassiopeia/preprocess/map_utils.pyx"],
    ),
    Extension(
        "cassiopeia.preprocess.collapse_cython",
        ["cassiopeia/preprocess/collapse_cython.pyx"],
    ),
    Extension(
        "cassiopeia.solver.ilp_solver_utilities",
        ["cassiopeia/solver/ilp_solver_utilities.pyx"],
    ),
]


setup(
    name="cassiopeia-lineage",
    python_requires='>=3.6',
    ext_modules=cythonize(to_cythonize),
    # ext_modules=to_cythonize,
    setup_requires=["cython", "numpy"],
    cmdclass=cmdclass,
    entry_points={"console_scripts": ["scLT = cassiopeia.__main__:main"]},
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
    version="1.0.4",
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
)
