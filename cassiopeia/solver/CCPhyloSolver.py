"""
This file stores a subclass of CassiopeiaSolver, the CCPhyloSolver. This is 
a wrapper around optimized agglomerative algorithms implemented by CCPhylo
(https://bitbucket.org/genomicepidemiology/ccphylo/src/master/). Methods
that will inherit from this class by default are DynamicNeighborJoiningSolver,
HeuristicNeighborJoiningSolver, and OptimizedUPGMASolver.
"""

import abc
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import os
import tempfile
import ete3
import subprocess
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import CassiopeiaSolver, solver_utilities


class CCPhyloSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    Distance based solver class.

    This solver serves as a wrapper around CCPhylo algorithms.

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
        method: str = "nj",
    ):

        super().__init__(prior_transformation)

        self.dissimilarity_function = dissimilarity_function
        self.add_root = add_root
        self.method = method

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

        dissimilarity_map = self.get_dissimilarity_map(cassiopeia_tree, layer)

        with tempfile.TemporaryDirectory() as temp_dir:

            # save dissimilarity map as phylip file
            dis_path = os.path.join(temp_dir, "dist.phylip")
            tree_path = os.path.join(temp_dir, "tree.nwk")
            solver_utilities.save_dissimilarity_as_phylip(dissimilarity_map, dis_path)

            # run ccphylo
            command = f". ~/.bashrc && ccphylo tree -i {dis_path} -o {tree_path} -m {self.method}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            T = ete3.Tree(tree_path, format=1)

            # remove temporary files
            os.remove(dis_path)
            os.remove(tree_path)

        # remove root from character matrix before populating tree
        if (cassiopeia_tree.root_sample_name
            in cassiopeia_tree.character_matrix.index):
            cassiopeia_tree.character_matrix = (
                cassiopeia_tree.character_matrix.drop(
                    index=cassiopeia_tree.root_sample_name
                )
            )

        # root tree
        if (self.add_root):
            T.set_outgroup(T.get_midpoint_outgroup())
            root = ete3.TreeNode(name="root")
            root.add_child(T)
    
        # populate tree
        T.ladderize(direction=1)
        cassiopeia_tree.populate_tree(T,layer=layer)
        print(stderr.decode("utf-8"))

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
        creating the dissimilarity map.

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

        if cassiopeia_tree.get_dissimilarity_map() is None:
            if self.dissimilarity_function is None:
                raise DistanceSolverError(
                    "Please specify a dissimilarity function or populate the "
                    "CassiopeiaTree object with a dissimilarity map"
                )

            cassiopeia_tree.compute_dissimilarity_map(
                self.dissimilarity_function, self.prior_transformation, layer
            )

