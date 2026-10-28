"""Build the Cython / C++ extension modules.

The ``.pyx`` files and C++ sources are the single source of truth and are
shipped in the sdist (see ``MANIFEST.in``); they are compiled at build time.
Cython and NumPy are guaranteed to be present because they are declared in
``[build-system].requires`` in ``pyproject.toml``, so an isolated PEP 517 build
(``pip install``/``pip wheel``/``python -m build``) works from either the source
tree or the sdist without any pre-generated C sources.
"""

import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# (fully-qualified module name, path to its .pyx relative to the project root)
_CYTHON_MODULES = [
    ("cassiopeia.solver.ilp_solver_utilities", "src/cassiopeia/solver/ilp_solver_utilities.pyx"),
    ("cassiopeia.solver.nj_solver_utilities", "src/cassiopeia/solver/nj_solver_utilities.pyx"),
    (
        "cassiopeia.solver.greedy_solver_utilities",
        "src/cassiopeia/solver/greedy_solver_utilities.pyx",
    ),
]

# Preprocessing kernels: C++20 sources compiled into one extension. Only zlib
# is required (BAM/BGZF reading and writing is implemented in-tree).
_NATIVE_DIR = "src/cassiopeia/preprocess/_native"
_NATIVE_SOURCES = [
    f"{_NATIVE_DIR}/_core.pyx",
    f"{_NATIVE_DIR}/src/bam_io.cpp",
    f"{_NATIVE_DIR}/src/sw.cpp",
    f"{_NATIVE_DIR}/src/caller.cpp",
    f"{_NATIVE_DIR}/src/alleles.cpp",
    f"{_NATIVE_DIR}/src/align.cpp",
    f"{_NATIVE_DIR}/src/intbc.cpp",
]
if sys.platform == "win32":
    _cpp_args = ["/std:c++20", "/O2"]
    _link_args = []
else:
    _cpp_args = ["-std=c++20", "-O3", "-pthread"]
    _link_args = ["-pthread"]

ext_modules = cythonize(
    [
        Extension(
            name,
            sources=[source],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        for name, source in _CYTHON_MODULES
    ]
    + [
        Extension(
            "cassiopeia.preprocess._native._core",
            sources=_NATIVE_SOURCES,
            include_dirs=[numpy.get_include(), _NATIVE_DIR],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c++",
            extra_compile_args=_cpp_args,
            extra_link_args=_link_args,
            libraries=[] if sys.platform == "win32" else ["z"],
        )
    ],
    language_level="3",
)

setup(ext_modules=ext_modules)
