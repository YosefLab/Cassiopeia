import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension

import numpy
from Cython.Build import cythonize

# https://github.com/mdgoldberg/poetry-cython-example


def build():
    extensions = [
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
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": 3},
    )

    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = "extended"

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()