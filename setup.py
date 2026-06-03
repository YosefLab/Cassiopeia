import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# collapse_cython ships a pre-generated .c source; the solver extensions are
# cythonized from their .pyx sources at build time.
ext_modules = [
    Extension(
        "cassiopeia.preprocess.collapse_cython",
        sources=["src/cassiopeia/preprocess/collapse_cython.c"],
        include_dirs=[numpy.get_include()],
        language="c",
    ),
]

ext_modules += cythonize(
    [
        Extension(
            "cassiopeia.solver.ilp_solver_utilities",
            sources=["src/cassiopeia/solver/ilp_solver_utilities.pyx"],
            include_dirs=[numpy.get_include()],
            language="c",
        ),
        Extension(
            "cassiopeia.solver.nj_solver_utilities",
            sources=["src/cassiopeia/solver/nj_solver_utilities.pyx"],
            include_dirs=[numpy.get_include()],
            language="c",
        ),
    ],
    language_level="3",
)

setup(
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
