import numpy
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "cassiopeia.preprocess.collapse_cython",
        sources=["cassiopeia/preprocess/collapse_cython.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
    Extension(
        "cassiopeia.solver.ilp_solver_utilities",
        sources=["cassiopeia/solver/ilp_solver_utilities.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
]

setup(
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
