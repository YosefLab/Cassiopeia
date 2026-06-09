"""Top level for Tree Solver development."""

from . import nj_solver_utilities, rooting
from .deprecated import (
    MaxCutGreedySolver,
    MaxCutSolver,
    PercolationSolver,
    SharedMutationJoiningSolver,
    SpectralGreedySolver,
    SpectralNeighborJoiningSolver,
    SpectralSolver,
)
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
