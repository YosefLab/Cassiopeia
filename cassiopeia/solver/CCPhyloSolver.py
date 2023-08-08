"""
This file stores a subclass of CassiopeiaSolver, the CCPhyloSolver. This is 
a wrapper around optimized agglomerative algorithms implemented by CCPhylo
(https://bitbucket.org/genomicepidemiology/ccphylo/src/master/). Methods
that will inherit from this class by default are DynamicNeighborJoiningSolver,
HeuristicNeighborJoiningSolver, NeighborJoiningSolver and UPGMASolver.
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
import configparser

from cassiopeia.data import CassiopeiaTree, utilities
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import (
    DistanceSolver,
    dissimilarity_functions,
    solver_utilities,
)


class CCPhyloSolver(DistanceSolver.DistanceSolver):
    """
    Distance based solver class. This solver serves as a wrapper around 
    CCPhylo algorithms.

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
        ] = dissimilarity_functions.weighted_hamming_distance,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
        fast = True, 
        method: str = "nj",
    ):

        super().__init__(prior_transformation)

        self.dissimilarity_function = dissimilarity_function
        self.add_root = add_root
        self.fast = fast
        self.method = method
        
    def _setup_ccphylo(self) -> None:

        # get ccphylo path
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__),"..","config.ini"))
        self.ccphylo_path = config.get("Paths","ccphylo_path")

        #check that ccphylo_path is valid
        if not os.path.exists(self.ccphylo_path):
            raise DistanceSolverError(
                f"ccphylo_path {self.ccphylo_path} does not exist. To use fast "
                "versions of Neighbor-Joining and UPGMA please install "
                "CCPhylo (https://bitbucket.org/genomicepidemiology/ccphylo/src/master/) "
                "set the ccphylo_path in the config.ini file then reinstall Cassiopeia."
            )
        
        #check that ccphylo_path is executable
        if not os.access(self.ccphylo_path, os.X_OK):
            raise DistanceSolverError(
                f"ccphylo_path {self.ccphylo_path} is not executable. To use fast "
                "versions of Neighbor-Joining and UPGMA please install "
                "CCPhylo (https://bitbucket.org/genomicepidemiology/ccphylo/src/master/) "
                "set the ccphylo_path in the config.ini file then reinstall Cassiopeia."
            )
    
    def _fast_solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solves a tree using fast distance-based algorithms implemented by 
        CCPhylo. To call this method the CCPhlyo package must be installed 
        and the ccphylo_path must be set in the config file. The method attribute
        specifies which algorithm to use. The function will update the `tree`.

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

        self._setup_ccphylo()

        dissimilarity_map = self.get_dissimilarity_map(cassiopeia_tree, layer)

        with tempfile.TemporaryDirectory() as temp_dir:

            # save dissimilarity map as phylip file
            dis_path = os.path.join(temp_dir, "dist.phylip")
            tree_path = os.path.join(temp_dir, "tree.nwk")
            solver_utilities.save_dissimilarity_as_phylip(dissimilarity_map, dis_path)

            # run ccphylo
            command = f"{self.ccphylo_path} tree -i {dis_path} -o {tree_path} -m {self.method}"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            T = ete3.Tree(tree_path, format=1)

            # remove temporary files
            os.remove(dis_path)
            os.remove(tree_path)

        # Covert to networkx
        tree = nx.Graph()
        internal_node_iter = 0
        for n in T.traverse():
            if n.name == "":
                n.name = f"cassiopeia_internal_node{internal_node_iter}"
                internal_node_iter += 1
            if not n.is_root():
                tree.add_edge(n.up.name,n.name)

        # find last split
        midpoint = T.get_midpoint_outgroup()
        root = T.get_tree_root()
        if midpoint in root.children:
            last_split = [root.name,midpoint.name]
        else:
            last_split = [root.name,root.children[0].name]
        tree.remove_edge(last_split[0],last_split[1])

        # root tree
        tree = self.root_tree(tree,cassiopeia_tree.root_sample_name,last_split)

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

        # populate tree
        cassiopeia_tree.populate_tree(tree,layer=layer)
        cassiopeia_tree.collapse_unifurcations()

        # collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solves a tree for a general bottom-up distance-based solver routine.
        If fast is set to True, this function will use the fast_solve method which
        is wrapper around CCPhylo algorithms. Otherwise, it will default to the 
        generic solve method in DistanceSolver. The function will update the `tree`
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

        if self.fast:
            self._fast_solve(cassiopeia_tree, layer, collapse_mutationless_edges, logfile)
        else:
            if self.__class__ == CCPhyloSolver:
                raise NotImplementedError(
                    "CCPhyloSolver does not implement solve. Please set fast to True" 
                    "to use the fast_solve method. Or use a subclass of CCPhyloSolver."
                )
            super().solve(cassiopeia_tree, layer, collapse_mutationless_edges, logfile)