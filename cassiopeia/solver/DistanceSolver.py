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
from typing import Callable, Dict, List, Optional, Tuple

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import CassiopeiaSolver


class DistanceSolveError(Exception):
    """An Exception class for all DistanceSolver subclasses."""

    pass


class DistanceSolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(self, dissimilarity_function: Optional[Callable] = None):

        super().__init__(None, None, None)

        self.dissimilarity_function = dissimilarity_function

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        dissimilarity_map: Optional[pd.DataFrame] = None,
        root_sample: Optional[str] = None,
        root_tree: bool = False,
    ) -> None:
        """A general bottom-up distance-based solver routine.

        The general solver routine proceeds by iteratively finding pairs of
        samples to join together into a "cherry" and then reform the
        dissimilarity matrix with respect to this new cherry. The implementation
        of how to find cherries and update the dissimilarity map is left to
        subclasses of DistanceSolver. The function will update the `tree`
        attribute of the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree object to be populated
            dissimilarity_map: Dissimilarity map storing the distances
                between samples
            root_sample: Sample to treat as a root
            root_tree: Whether or not to root the tree after the routine
        """

        (
            unique_character_matrix,
            dissimilarity_map,
            state_to_sample_mapping,
            root_sample,
        ) = self.setup_solver(
            cassiopeia_tree, dissimilarity_map, root_sample, root_tree
        )

        N = dissimilarity_map.shape[0]

        identifier_to_sample = dict(
            zip([str(i) for i in range(N)], dissimilarity_map.index)
        )

        # instantiate a dissimilarity map that can be updated as we join
        # together nodes.
        _dissimilarity_map = dissimilarity_map.copy()

        # instantiate a tree where all samples appear as leaves.
        tree = nx.Graph()
        tree.add_nodes_from(dissimilarity_map.index)

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
        tree = self.append_sample_names(
            tree, state_to_sample_mapping
        )

        if root_sample is not None:
            tree = self.root_tree(tree, root_sample)

        cassiopeia_tree.populate_tree(tree)

    def setup_solver(
        self,
        cassiopeia_tree: CassiopeiaTree,
        dissimilarity_map: Optional[pd.DataFrame] = None,
        root_sample: Optional[str] = None,
        root_tree: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], str]:
        """Sets up the solver.

        Sets up the solver with respect to the input CassiopeiaTree by
        extracting the character matrix, creating the dissimilarity map if
        needed, and setting up the "root" sample if the tree will be rooted.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`.
            dissimilarity_map: Dissimilarity map. If this is None, the function
                will create it using the dissimilarity function specified.
            root_sample: Sample to treat as the root.
            root_tree: Whether or not to root the tree after inference.

        Returns:
            A character matrix with duplicate rows filtered out, a
                dissimilarity map, a mapping from state to sample name, and
                the sample to treat as a root.
        """

        if self.dissimilarity_function is None and dissimilarity_map is None:
            raise DistanceSolveError(
                "Please specify a dissimilarity map or dissimilarity function"
            )

        character_matrix = (
            cassiopeia_tree.get_original_character_matrix().copy()
        )            

        if root_sample is None and root_tree:

            if self.dissimilarity_function is None:
                raise DistanceSolveError(
                    "Please specify a root sample or provide a dissimilarity "
                    "function by which to add a root to the dissimilarity map"
                )

            root = [0] * character_matrix.shape[1]
            character_matrix.loc["root"] = root
            root_sample = "root"

            # if root sample is not specified, we'll add the implicit root
            # and recompute the dissimilarity map
            dissimilarity_map = None

            root_sample = "root"

        unique_character_matrix = character_matrix.drop_duplicates().copy()
        unique_character_matrix.index = [
            f"state{i}" for i in range(unique_character_matrix.shape[0])
        ]

        # Create the dissimilarity map if not specified
        if dissimilarity_map is None:
            N = unique_character_matrix.shape[0]
            dissimilarity_map = self.compute_dissimilarity_map(
                unique_character_matrix.to_numpy(),
                N,
                cassiopeia_tree.priors,
                cassiopeia_tree.missing_state_indicator,
            )
            dissimilarity_map = scipy.spatial.distance.squareform(
                dissimilarity_map
            )

            dissimilarity_map = pd.DataFrame(
                dissimilarity_map,
                index=unique_character_matrix.index,
                columns=unique_character_matrix.index,
            )

        # create state to sample name mapping
        state_to_sample_mapping = {}
        state_lookup = character_matrix.apply(lambda x: tuple(x.values), axis=1)
        for state in unique_character_matrix.index:
            characters = tuple(unique_character_matrix.loc[state].values)
            samples = state_lookup[state_lookup == characters].index.values
            state_to_sample_mapping[state] = samples

        return (
            unique_character_matrix,
            dissimilarity_map,
            state_to_sample_mapping,
            root_sample
        )

    @numba.jit(forceobj=True)
    def compute_dissimilarity_map(
        self,
        cm: np.array,
        C: int,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        missing_indicator: int = -1,
    ) -> np.array:
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
                    s1, s2, priors, missing_indicator
                )
                k += 1

        return dm

    @abc.abstractmethod
    def root_tree(self):
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

    def append_sample_names(
        self,
        solution: nx.DiGraph,
        state_to_sample_mapping: Dict[str, List[str]],
    ) -> nx.DiGraph:
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

        for l in leaves:

            if l not in state_to_sample_mapping.keys():
                continue

            # remove samples with the same name as the leaf
            samples = [s for s in state_to_sample_mapping[l] if s != l]

            if len(samples) > 0:
                solution.add_edges_from([(l, sample) for sample in samples])

        return solution
