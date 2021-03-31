"""Top level for Tree Solver development."""

__author__ = "Matt Jones, Alex Khodaverdian"
__email__ = "mattjones315@berkeley.edu"
__version__ = "0.0.1"

from .HybridSolver import HybridSolver
from .ILPSolver import ILPSolver
from .MaxCutGreedySolver import MaxCutGreedySolver
from .MaxCutSolver import MaxCutSolver
from .NeighborJoiningSolver import NeighborJoiningSolver
from .solver_utilities import collapse_tree
from .SpectralGreedySolver import SpectralGreedySolver
from .SpectralSolver import SpectralSolver
from .VanillaGreedySolver import VanillaGreedySolver
from . import dissimilarity_functions as dissimilarity
