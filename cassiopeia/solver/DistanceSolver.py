
"""
This file stores a subclass of CassiopeiaSolver, the DistanceSolver. Generally,
the inference procedures that inherit from this method will need to implement
methods for selecting "cherries" and updating the dissimilarity map. Methods
that will inherit from this class by default are Neighbor-Joining and UPGMA.
There may be other subclasses of this
"""
import abc
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple

from cassiopeia.solver import CassiopeiaSolver


class DistanceSolveError(Exception):
    """An Exception class for all DistanceSolver subclasses.
    """
    pass


class DistanceSolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        dissimilarity_map: Optional[pd.DataFrame] = None,
        dissimilarity_function: Optional[Callable] = None,
    ):

        if dissimilarity_function is None and dissimilarity_map is None:
            raise DistanceSolveError(
                "Please specify a dissimilarity map or dissimilarity function"
            )

        super().__init__(character_matrix, meta_data, priors)

        self.tree = None
        
        self.dissimilarity_map = dissimilarity_map
        self.dissimilarity_function = dissimilarity_function

        # Create the dissimilarity map if not specified
        if self.dissimilarity_map is None:

            N = self.character_matrix.shape[0]
            dissimilarity_map = np.zeros((N, N))
            for i in range(character_matrix.shape[0]):

                ind1 = self.character_matrix.iloc[i, :].values

                for j in range(i + 1, character_matrix.shape[0]):

                    ind2 = self.character_matrix.iloc[j, :].values
                    dissimilarity_map[i, j] = dissimilarity_map[
                        j, i
                    ] = self.dissimilarity_function(ind1, ind2)

            self.dissimilarity_map = pd.DataFrame(
                dissimilarity_map,
                index=self.character_matrix.index,
                columns=self.character_matrix.index,
            )

    def solve(self):
        """A general bottom-up distance-based solver routine.

        The general solver routine proceeds by iteratively finding pairs of
        sapmles to join together into a "cherry" and then reform the
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
        self.tree = tree

        self.root_tree()

    @abc.abstractmethod
    def root_tree(self):
        """Roots a tree.

        Finds a location on the tree to place a root and converts the general
        graph to a directed graph with respect to that root.
        """
        pass

    @abc.abstractmethod
    def find_cherry(self, dissimilarity_map: np.array) -> Tuple[int, int]:
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
