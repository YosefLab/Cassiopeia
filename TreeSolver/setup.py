from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules=cythonize(["lineage_solver/solver_utils.pyx", "simulation_tools/dataset_generation.pyx"]),
)
