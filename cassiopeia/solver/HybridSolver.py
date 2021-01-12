"""
This file stores a subclass of CassiopeiaSolver, the HybridSolver. This solver
consists of two sub-solvers -- a top (greedy) and a bottom solver. The greedy
solver will be applied until a certain criteria is reached (be it a maximum LCA 
distance or a number of cells) and then another solver is employed to
infer relationships at the bottom of the phylogeny under construction.

In Jones et al, the Cassiopeia-Hybrid algorithm is a HybridSolver that consists
of a VanillaGreedySolver stacked on top of a ILPSolver instance.
"""
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import GreedySolver
from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import solver_utilities


class HybridSolverError(Exception):
    """An Exception class for all HybridSolver subclasses."""

    pass

class HybridSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    HybridSolver is an class representing the structure of Cassiopeia Hybrid
    inference algorithms. The solver procedure contains logic for building tree
    starting with a top-down greedy algorithm until a predetermined criteria is reached
    at which point a more complex algorithm is used to reconstruct each
    subproblem. The top-down algorithm _must_ be a subclass of a GreedySolver 
    as it must have functions `find_split` and `perform_split`. The solver
    employed at the bottom of the tree can be any CassiopeiaSolver subclass and
    need only have a `solve` method.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        top_solver: An algorithm to be applied at the top of the tree. Must
            be a subclass of GreedySolver.
        bottom_solver: An algorithm to be applied at the bottom of the tree.
            Must be a subclass of CassiopeiaSolver.
        lca_cutoff: Distance to the latest-common-ancestor (LCA) of a subclade
            to be used as a cutoff for transitioning to the bottom solver.
        cell_cutoff: Number of cells in a subclade to be used as a cutoff
            for transitioning to the bottom solver.
        threads: Number of threads to be used. This corresponds to the number of
            subproblems to be run concurrently with the bottom solver.
    """
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        missing_char: int = -1,
        top_solver: GreedySolver.GreedySolver,
        bottom_solver: CassiopeiaSolver.CassiopeiaSolver,
        lca_cutoff: float = None,
        cell_cutoff: int = None,
        threads: int = 1,
    ):

        if lca_cutoff is None and cell_cutoff is None:
            raise HybridSolverError(
                "Please specify a cutoff, either through lca_cutoff or cell_cutoff"
            )

        super().__init__(character_matrix, missing_char, meta_data, priors)

        self.lca_cutoff = lca_cutoff
        self.cell_cutoff = cell_cutoff

        self.threads = threads

        self.tree = None

    def solve(self):
        """The general hybrid solver routine.

        The hybrid solver proceeds by clustering together cells using the 
        algorithm stored in the top_solver until a criteria is reached. Once
        this criteria is reached, the bottom_solver is applied to each
        subproblem left over from the "greedy" clustering.
        """            

        _, subproblems = self.__apply_top_solver(None, list(self.unique_character_matrix.index))

        # implement multi-threaded bottom solver approach

        # Collapse 0-mutation edges and append duplicate samples
        self.tree = solver_utilities.collapse_tree(
            self.tree, True, self.unique_character_matrix, self.missing_char
        )
        self.tree = self.add_duplicates_to_tree(self.tree)

    def __apply_top_solver(self, root: Optional[int] = None, samples: List[int]) -> int, List[Tuple[int], List[int]]:
        """Applies the top solver to samples.

        A private helper method for applying the top solver to the samples
        until a criteria is hit. Subproblems and the root ID are returned.

        Args:
            root: Node ID of the root in the subtree containing the samples.
            samples: Samples in the subclade.

        Returns:
            The ID of the node serving as the root of the tree containing the 
                samples, and a list of subproblems in the form
                [subtree-root, subtree-samples].
        """

        mutation_frequences = self.top_solver.compute_mutation_frequencies(samples)

        clades = list(self.perform_split(mutation_frequencies, samples))
        root = (
                len(self.tree.nodes)
                - self.unique_character_matrix.shape[0]
                + self.character_matrix.shape[0]
            )
        self.tree.add_node(root)

        subproblems = []
        for clade in clades:

            if len(clade) == 0:
                clades.remove(clade)

            if self.assess_cutoff(clade):
                subproblems += [root, clade]
                clades.remove(clade)

        if len(clades) == 1:
            for clade in clades[0]:
                self.tree.add_edge(root, clade)
            return root, []

        for clade in clades:
            child, new_subproblems = self.__apply_top_solver(clade)
            self.tree.add_edge(root, child)

            subproblems += new_subproblems

        return root, subproblems

    def assess_cutoff(self, samples: List[int]) -> bool:
        """Assesses samples with respect to hybrid cutoff.

        Args:
            samples: A list of samples in a clade.

        Returns:
            True if the cutoff is reached, False if not. 
        """
        
        if self.cell_cutoff is None:
            root_states = (
                solver_utilities.get_lca_characters(
                    self.unique_character_matrix.values.tolist(), self.missing_char
                )

            lca_distances = [
                dissimilarity_functions.hamming_distance(root_states, np.array(u))
                for u in samples
            ]

            if np.max(lca_distances) >= self.lca_cutoff:
                return True
            
        else:
            if len(samples) >= self.cell_cutoff:
                return return True

        return False