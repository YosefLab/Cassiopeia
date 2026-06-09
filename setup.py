"""Build the Cython extension modules.

The ``.pyx`` files are the single source of truth and are shipped in the sdist
(see ``MANIFEST.in``); they are cythonized to C and compiled at build time.
Cython and NumPy are guaranteed to be present because they are declared in
``[build-system].requires`` in ``pyproject.toml``, so an isolated PEP 517 build
(``pip install``/``pip wheel``/``python -m build``) works from either the source
tree or the sdist without any pre-generated C sources.
"""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# (fully-qualified module name, path to its .pyx relative to the project root)
_CYTHON_MODULES = [
    ("cassiopeia.preprocess.collapse_cython", "src/cassiopeia/preprocess/collapse_cython.pyx"),
    ("cassiopeia.solver.ilp_solver_utilities", "src/cassiopeia/solver/ilp_solver_utilities.pyx"),
    ("cassiopeia.solver.nj_solver_utilities", "src/cassiopeia/solver/nj_solver_utilities.pyx"),
    (
        "cassiopeia.solver.greedy_solver_utilities",
        "src/cassiopeia/solver/greedy_solver_utilities.pyx",
    ),
]

ext_modules = cythonize(
    [
        Extension(
            name,
            sources=[source],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        for name, source in _CYTHON_MODULES
    ],
    language_level="3",
)

setup(ext_modules=ext_modules)
