#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from setuptools import setup, Extension, find_packages
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext


with open("README.md") as readme_file:
    readme = readme_file.read()


requirements = [
        "numpy > 1.17",
        "matplotlib >= 2.2.2",
        "pandas >= 0.22.0",
        "networkx >= 2.0",
        "tqdm >= 4",
        # "gurobipy",
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
        'nbformat >= 4.4.0',
        'hits',
        'scikit-bio >= 0.5.5'
]


author = "Matthew Jones, Alex Khodaverdian, Jeffrey Quinn, Jeffrey Hussmann, Michelle Chan"

cmdclass = {'build_ext': build_ext}

# files to wrap with cython
to_cythonize = [Extension("cassiopeia.solver.lineage_solver.solver_utils", ["cassiopeia/solver/lineage_solver/solver_utils.pyx"]),
                Extension("cassiopeia.solver.simulation_tools.dataset_generation", ["cassiopeia/solver/simulation_tools/dataset_generation.pyx"]),
                Extension("cassiopeia.preprocess.lineageGroup_utils", ["cassiopeia/preprocess/lineageGroup_utils.pyx"]), 
                Extension("cassiopeia.preprocess.collapse_cython", ["cassiopeia/preprocess/collapse_cython.pyx"]),
                Extension("cassiopeia.solver.utilities", ['cassiopeia/solver/utilities.pyx'])] 
#to_cythonize = [Extension("TreeSolver.lineage_solver.solver_utils", ["cassiopeia/TreeSolver/lineage_solver/solver_utils.pyx"]),
#               Extension("TreeSolver.simulation_tools.dataset_generation", ["cassiopeia/TreeSolver/simulation_tools/dataset_generation.pyx"]),
#               Extension("ProcessingPipeline.process.lineageGroup_utils", ["cassiopeia/ProcessingPipeline/process/lineageGroup_utils.pyx"]), 
#               Extension("ProcessingPipeline.process.collapse_cython", ["cassiopeia/ProcessingPipeline/process/collapse_cython.pyx"])] 
                

setup(
        name="cassiopeia-lineage",
        ext_modules=cythonize(to_cythonize),
        #ext_modules=to_cythonize,
        setup_requires=['cython', 'numpy'],
        cmdclass=cmdclass,
        entry_points={
            'console_scripts': ['scLT = cassiopeia.__main__:main',
                                'reconstruct-lineage = cassiopeia.solver.reconstruct_tree:main',
                                'post-process-tree = cassiopeia.solver.post_process_tree:main',
                                'stress-test = cassiopeia.solver.reconstruct_sim_tree:main',
                                'simulate-tree = cassiopeia.solver.simulate_tree:main',
                                'call-lineages = cassiopeia.preprocess.lineageGroup:main',
                                'filter-molecule-table = cassiopeia.preprocess.filterMoleculeTables:main']
            
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
        version='1.0.4',
        zip_safe=False,
        test_suite='nose.collector',
        test_require=['nose']
)
