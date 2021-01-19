"""
This file stores a subclass of CassiopeiaSolver, the DistanceSolver. Generally,
the inference procedures that inherit from this method will need to implement
methods for selecting "cherries" and updating the dissimilarity map. Methods
that will inherit from this class by default are Neighbor-Joining and UPGMA.
There may be other subclasses of this
"""
import abc
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy
from typing import Callable, Dict, Optional, Tuple

from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import solver_utilities


class DistanceSolveError(Exception):
    """An Exception class for all DistanceSolver subclasses."""

    pass


class DistanceSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    DistanceSolver is an abstract class representing the structure of bottom-up
    inference algorithms. The solver procedure contains logic to build a tree
    from the leaves by iteratively joining the set of samples based on their
    distances from each other, defined by a provided dissimilarity map or
    function. Each subclass will implement "find_cherry" and
    "update_dissimilarity_map", which are the procedures for finding the next
    cherry to join and updating the dissimilarity map, respectively.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_function: A function defining a transformation on the priors
            in forming weights
        dissimilarity_map: A dissimilarity map describing the distances between
            samples.
        dissimilarity_function: A function by which to compute the dissimilarity
            map. Optional if a dissimilarity map is already provided.

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors
        dissimilarity_map: Dissimilarity map describing distances between
            samples
        dissimilarity_function: Function to compute the dissimilarity between
            samples.
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        unique_character_matrix: A character matrix with duplicate rows filtered out
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: int = -1,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        prior_function: Optional[Callable[[float], float]] = None,
        dissimilarity_map: Optional[pd.DataFrame] = None,
        dissimilarity_function: Optional[Callable] = None,
    ):

        if dissimilarity_function is None and dissimilarity_map is None:
            raise DistanceSolveError(
                "Please specify a dissimilarity map or dissimilarity function"
            )

        super().__init__(character_matrix, meta_data, priors)

        self.tree = None

        if priors:
            if prior_function:
                self.weights = solver_utilities.transform_priors(
                    priors, prior_function
                )
            else:
                self.weights = solver_utilities.transform_priors(
                    priors, lambda x: -np.log(x)
                )
        else:
            self.weights = None

        self.dissimilarity_map = dissimilarity_map
        self.dissimilarity_function = dissimilarity_function

        self.unique_character_matrix = (
            self.character_matrix.drop_duplicates().copy()
        )
        self.unique_character_matrix.index = [
            f"state{i}" for i in range(self.unique_character_matrix.shape[0])
        ]

        # Create the dissimilarity map if not specified
        if self.dissimilarity_map is None:
            N = self.unique_character_matrix.shape[0]
            dissimilarity_map = self.compute_dissimilarity_map(
                self.unique_character_matrix.to_numpy(), N
            )
            dissimilarity_map = scipy.spatial.distance.squareform(
                dissimilarity_map
            )

            self.dissimilarity_map = pd.DataFrame(
                dissimilarity_map,
                index=self.unique_character_matrix.index,
                columns=self.unique_character_matrix.index,
            )

    def solve(self):
        """A general bottom-up distance-based solver routine.

        The general solver routine proceeds by iteratively finding pairs of
        samples to join together into a "cherry" and then reform the
        dissimilarity matrix with respect to this new cherry. The implementation
        of how to find cherries and update the dissimilarity map is left to
        subclasses of DistanceSolver. The function by default updates the
        self.tree instance variable.
        """

        N = self.dissimilarity_map.shape[0]

        identifier_to_sample = dict(
            zip([str(i) for i in range(N)], self.dissimilarity_map.index)
        )

        # instantiate a dissimilarity map that can be updated as we join
        # together nodes.
        _dissimilarity_map = self.dissimilarity_map.copy()

        # instantiate a tree where all samples appear as leaves.
        tree = nx.Graph()
        tree.add_nodes_from(self.dissimilarity_map.index)

        while N > 2:

            i, j = self.find_cherry(_dissimilarity_map.values)

            # get indices in the dissimilarity matrix to join
            node_i, node_j = (
                _dissimilarity_map.index[i],
                _dissimilarity_map.index[j],
            )

            new_node_name = str(len(tree.nodes))
            tree.add_node(new_node_name)
            tree.add_edges_from(
                [(new_node_name, node_i), (new_node_name, node_j)]
            )

            _dissimilarity_map = self.update_dissimilarity_map(
                _dissimilarity_map, (node_i, node_j), new_node_name
            )

            N = _dissimilarity_map.shape[0]

        remaining_samples = _dissimilarity_map.index.values
        tree.add_edge(remaining_samples[0], remaining_samples[1])

        tree = nx.relabel_nodes(tree, identifier_to_sample)
        tree = self.append_sample_names(tree)
        tree = self.root_tree(tree)

        self.tree = tree

    @numba.jit(forceobj=True)
    def compute_dissimilarity_map(self, cm: np.array, C: int) -> np.array:
        """Compute the dissimilarity between all samples

        An optimized function for computing pairwise dissimilarities between
        samples in a character matrix according to the dissimilarity function.

        Args:
            cm: Character matrix
            C: Number of samples
            delta: A dissimilarity function that takes in two arrays and returns
                a dissimilarity

        Returns:
            A dissimilarity mapping as a flattened array.
        """

        dm = np.zeros(C * (C - 1) // 2, dtype=float)
        k = 0
        for i in range(C - 1):
            for j in range(i + 1, C):

                s1 = cm[i, :]
                s2 = cm[j, :]

                dm[k] = self.dissimilarity_function(
                    s1, s2, self.missing_char, self.weights
                )
                k += 1

        return dm

    @abc.abstractmethod
    def root_tree(self, tree):
        """Roots a tree.

        Finds a location on the tree to place a root and converts the general
        graph to a directed graph with respect to that root.
        """
        pass

    @abc.abstractmethod
    def find_cherry(
        self, dissimilarity_map: np.array(float)
    ) -> Tuple[int, int]:
        """Selects two samples to join together as a cherry.

        Selects two samples from the dissimilarity map to join together as a
        cherry in the forming tree.

        Args:
            dissimilarity_map: A dissimilarity map relating samples

        Returns:
            A tuple of samples to join together.
        """
        pass

    @abc.abstractmethod
    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame,
        cherry: Tuple[str, str],
        new_node: str,
    ) -> pd.DataFrame:
        """Updates dissimilarity map with respect to new cherry.

        Args:
            dissimilarity_map: Dissimilarity map to update
            cherry1: One of the children to join.
            cherry2: One of the children to join.
            new_node: New node name to add to the dissimilarity map

        Returns:
            An updated dissimilarity map.
        """
        pass

    def append_sample_names(self, solution: nx.DiGraph) -> nx.DiGraph:
        """Append sample names to character states in tree.

        Given a tree where every node corresponds to a set of character states,
        append sample names at the deepest node that has its character
        state. The DistanceSolver by default has observed samples as leaves,
        so this procedure is simply to stitch samples names onto the leaves
        at the appropriate location.

        Args:
            solution: A DistanceSolver solution that we wish to add sample
                names to.

        Returns:
            A solution with extra leaves corresponding to sample names.
        """

        leaves = [n for n in solution if solution.degree(n) == 1]

        sample_lookup = self.character_matrix.apply(
            lambda x: tuple(x.values), axis=1
        )

        for l in leaves:

            if l not in self.unique_character_matrix.index:
                continue

            character_state = tuple(self.unique_character_matrix.loc[l].values)
            samples = sample_lookup[
                sample_lookup == character_state
            ].index.values

            # remove samples with the same name as the leaf
            samples = [s for s in samples if s != l]

            if len(samples) > 0:
                solution.add_edges_from([(l, sample) for sample in samples])

        return solution
