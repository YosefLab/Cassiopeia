#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, Distribution, find_packages
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "Biopython>=1.71",
    "bokeh>=0.12.15",
    "cython>=0.29.2",
    "ete3>=3.1.1",
    "hits",
    "itolapi",
    "matplotlib>=2.2.2",
    "nbconvert>=5.4.0",
    "nbformat>=4.4.0",
    "networkx>=2.5",
    "ngs-tools>=1.5.3",
    "numba>=0.51.0",
    "numpy>=1.19.5",
    "pandas>=1.1.4",
    "pysam>=0.14.1",
    "python-Levenshtein",
    "PyYAML>=3.12",
    "scipy>=1.2.0",
    "typing-extensions>=3.7.4",
    "tqdm>=4",
    "cvxpy",
    "parameterized",
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
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "cassiopeia.tools.branch_length_estimator.iid_exponential_bayesian_cpp",
        ["cassiopeia/preprocess/doublet_utils.pyx"],
    ),
]

extension_modules_with_custom_compile_args = [
    Extension(
        "cassiopeia.tools.branch_length_estimator._iid_exponential_bayesian",
        sources=[
            "cassiopeia/tools/branch_length_estimator/_iid_exponential_bayesian.pyx",
            "cassiopeia/tools/branch_length_estimator/_iid_exponential_bayesian_cpp.cpp",
        ],
        extra_compile_args=[
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-O3",
        ],
        language="c++",
    ),
]

setup(
    name="cassiopeia-lineage",
    python_requires=">=3.6",
    ext_modules=cythonize(
        to_cythonize, compiler_directives={"language_level": "3"}
    )
    + extension_modules_with_custom_compile_args,
    # ext_modules=to_cythonize,
    setup_requires=["cython", "numpy"],
    cmdclass=cmdclass,
    entry_points={
        "console_scripts": [
            "cassiopeia-preprocess = cassiopeia.preprocess.cassiopeia_preprocess:main"
        ]
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
    version="2.0.0",
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
)
