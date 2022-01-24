"""
This file stores a subclass of CassiopeiaSolver, the DistanceSolver. Generally,
the inference procedures that inherit from this method will need to implement
methods for selecting "cherries" and updating the dissimilarity map. Methods
that will inherit from this class by default are Neighbor-Joining and UPGMA.
There may be other subclasses of this.
"""
import abc
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import CassiopeiaSolver, solver_utilities


class DistanceSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    Distance based solver class.

    This solver serves as a generic Distance-based solver. Briefly, all of the
    classes that derive from this class will use a dissimilarity map to
    iteratively choose samples to merge and update the dissimilarity map
    based on this merging. An example of a derived class is the
    NeighborJoiningSolver which uses the Q-criterion to iteratively join
    samples until no samples remain.

    TODO(mgjones, sprillo, rzhang): Specify functions to use for rooting, etc.
        the trees that are produced via a DistanceSolver. Add compositional
        framework.

    TODO(mgjones, rzhang): Make the solver work with similarity maps as
        flattened arrays

    Args:
        dissimilarity_function: Function that can be used to compute the
            dissimilarity between samples.
        add_root: Whether or not to add an implicit root the tree. Only
            pertinent in algorithms that return an unrooted tree, by default
            (e.g. Neighbor Joining). Will not override an explicitly defined
            root, specified by the 'root_sample_name' attribute in the
            CassiopeiaTree
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        dissimilarity_function: Function used to compute dissimilarity between
            samples.
        add_root: Whether or not to add an implicit root the tree.
        prior_transformation: Function to use when transforming priors into
            weights.
    """

    def __init__(
        self,
        dissimilarity_function: Optional[
            Callable[
                [np.array, np.array, int, Dict[int, Dict[int, float]]], float
            ]
        ] = None,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(prior_transformation)

        self.dissimilarity_function = dissimilarity_function
        self.add_root = add_root

    def get_dissimilarity_map(
        self, 
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None
    ) -> pd.DataFrame: 
        """Obtains or generates a matrix that is updated throughout the solver.

        The highest-level method to obtain a dissimilarity map, which
        will be the matrix primarily used throughout the solve method. This
        matrix contains the pairwise dissimilarity between samples which is used
        for identifying sample pairs to merge, and will be updated at every
        iteration within the solve method. This method is not limited to
        outputting dissimilarity maps, but is instead deliberately
        designed to be overwritten to allow for use of similarity maps or other
        algorithm-specific sample to sample comparison maps in derived classes.

        Args:
            cassiopeia_tree: Tree object from which the 
                dissimilarity map is generated from
            layer: Layer storing the character matrix 
                for solving. If None, the default character matrix is used in 
                the CassiopeiaTree.

        Returns:
            pd.DataFrame: The matrix that will be used throughout the solve 
                method.
        """

        self.setup_dissimilarity_map(cassiopeia_tree, layer)
        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()

        return dissimilarity_map


    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solves a tree for a general bottom-up distance-based solver routine.

        The general solver routine proceeds by iteratively finding pairs of
        samples to join together into a "cherry" and then reform the
        dissimilarity matrix with respect to this new cherry. The implementation
        of how to find cherries and update the dissimilarity map is left to
        subclasses of DistanceSolver. The function will update the `tree`
        attribute of the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree object to be populated
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
            logfile: File location to log output. Not currently used.
        """
        node_name_generator = solver_utilities.node_name_generator()

        dissimilarity_map = self.get_dissimilarity_map(cassiopeia_tree, layer)

        N = dissimilarity_map.shape[0]

        # instantiate a dissimilarity map that can be updated as we join
        # together nodes.
        _dissimilarity_map = dissimilarity_map.copy()

        # instantiate a tree where all samples appear as leaves.
        tree = nx.Graph()
        tree.add_nodes_from(_dissimilarity_map.index)

        while N > 2:

            i, j = self.find_cherry(_dissimilarity_map.to_numpy())

            # get indices in the dissimilarity matrix to join
            node_i, node_j = (
                _dissimilarity_map.index[i],
                _dissimilarity_map.index[j],
            )

            new_node_name = next(node_name_generator)
            tree.add_node(new_node_name)
            tree.add_edges_from(
                [(new_node_name, node_i), (new_node_name, node_j)]
            )

            _dissimilarity_map = self.update_dissimilarity_map(
                _dissimilarity_map, (node_i, node_j), new_node_name
            )

            N = _dissimilarity_map.shape[0]

        tree = self.root_tree(
            tree,
            cassiopeia_tree.root_sample_name,
            _dissimilarity_map.index.values,
        )

        # remove root from character matrix before populating tree
        if (
            cassiopeia_tree.root_sample_name
            in cassiopeia_tree.character_matrix.index
        ):
            cassiopeia_tree.character_matrix = (
                cassiopeia_tree.character_matrix.drop(
                    index=cassiopeia_tree.root_sample_name
                )
            )

        cassiopeia_tree.populate_tree(tree, layer=layer)
        cassiopeia_tree.collapse_unifurcations()

        # collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

    def setup_dissimilarity_map(
        self, cassiopeia_tree: CassiopeiaTree, layer: Optional[str] = None
    ) -> None:
        """Sets up the solver.

        Sets up the solver with respect to the input CassiopeiaTree by
        creating the dissimilarity map if needed and setting up the
        "root" sample if the tree will be rooted. Operates directly on the
        CassiopeiaTree.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.

        Raises:
            A `DistanceSolverError` if rooting parameters are not passed in
                correctly (i.e. no root is specified and the user has not
                asked to find a root) or when a dissimilarity map cannot
                be found or computed.
        """

        # if root sample is not specified, we'll add the implicit root
        # and recompute the dissimilarity map

        if cassiopeia_tree.root_sample_name is None:
            if self.add_root:
                self.setup_root_finder(cassiopeia_tree)

            else:
                raise DistanceSolverError(
                    "Please specify an explicit root sample in the Cassiopeia Tree"
                    " or specify the solver to add an implicit root"
                )

        if cassiopeia_tree.get_dissimilarity_map() is None:
            if self.dissimilarity_function is None:
                raise DistanceSolverError(
                    "Please specify a dissimilarity function or populate the "
                    "CassiopeiaTree object with a dissimilarity map"
                )

            cassiopeia_tree.compute_dissimilarity_map(
                self.dissimilarity_function, self.prior_transformation, layer
            )

    @abc.abstractmethod
    def root_tree(
        self, tree: nx.Graph, root_sample: str, remaining_samples: List[str]
    ) -> nx.DiGraph:
        """Roots a tree.

        Finds a location on the tree to place a root and converts the general
        graph to a directed graph with respect to that root.

        Args:
            tree: an undirected networkx tree topology
            root_sample: node name to treat as the root of the tree topology
            remaining_samples: samples yet to be added to the tree.

        Returns:
            A rooted networkx tree
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

    @abc.abstractmethod
    def setup_root_finder(self, cassiopeia_tree: CassiopeiaTree) -> None:
        """Defines how an implicit root is to be added.

        Sets up the root sample for the tree solver, operating directly on the
        CassiopeiaTree.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`
        """
        pass
