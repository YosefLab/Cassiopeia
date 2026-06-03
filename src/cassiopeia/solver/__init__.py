"""Top level for Tree Solver development."""

from . import dissimilarity_functions, nj_solver_utilities, rooting
from .deprecated import (
    MaxCutGreedySolver,
    MaxCutSolver,
    PercolationSolver,
    SharedMutationJoiningSolver,
    SpectralGreedySolver,
    SpectralNeighborJoiningSolver,
    SpectralSolver,
)
from .dissimilarity import dissimilarity
from .greedy import VanillaGreedySolver, greedy
from .hybrid import HybridSolver, hybrid
from .ilp import ILPSolver, ilp
from .neighbor_joining import NeighborJoiningSolver, nj
from .rooting import reroot
from .upgma import UPGMASolver, upgma

__all__ = [
    # functional API
    "nj",
    "upgma",
    "greedy",
    "ilp",
    "hybrid",
    "reroot",
    # modules
    "rooting",
    "dissimilarity_functions",
    "dissimilarity",
    # backward-compat class shims (deprecated)
    "NeighborJoiningSolver",
    "UPGMASolver",
    "VanillaGreedySolver",
    "ILPSolver",
    "HybridSolver",
    # deprecated, removed-implementation solvers
    "MaxCutSolver",
    "MaxCutGreedySolver",
    "SpectralSolver",
    "SpectralGreedySolver",
    "SharedMutationJoiningSolver",
    "PercolationSolver",
    "SpectralNeighborJoiningSolver",
]
