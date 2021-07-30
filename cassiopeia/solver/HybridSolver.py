"""
This file stores a subclass of CassiopeiaSolver, the HybridSolver. This solver
consists of two sub-solvers -- a top (greedy) and a bottom solver. The greedy
solver will be applied until a certain criteria is reached (be it a maximum LCA
distance or a number of cells) and then another solver is employed to
infer relationships at the bottom of the phylogeny under construction.

In Jones et al, the Cassiopeia-Hybrid algorithm is a HybridSolver that consists
of a VanillaGreedySolver stacked on top of a ILPSolver instance.
"""
from typing import Dict, List, Generator, Optional, Tuple

import multiprocessing
import networkx as nx
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
from tqdm.auto import tqdm

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import HybridSolverError
from cassiopeia.solver import (
    CassiopeiaSolver,
    dissimilarity_functions,
    GreedySolver,
    solver_utilities,
)


class HybridSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    The Hybrid Cassiopeia solver.
    
    HybridSolver is an class representing the structure of Cassiopeia Hybrid
    inference algorithms. The solver procedure contains logic for building tree
    starting with a top-down greedy algorithm until a predetermined criteria is
    reached at which point a more complex algorithm is used to reconstruct each
    subproblem. The top-down algorithm _must_ be a subclass of a GreedySolver
    as it must have functions `find_split` and `perform_split`. The solver
    employed at the bottom of the tree can be any CassiopeiaSolver subclass and
    need only have a `solve` method.

    Args:
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
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p
    """

    def __init__(
        self,
        top_solver: GreedySolver.GreedySolver,
        bottom_solver: CassiopeiaSolver.CassiopeiaSolver,
        lca_cutoff: float = None,
        cell_cutoff: int = None,
        threads: int = 1,
        prior_transformation: str = "negative_log",
    ):

        if lca_cutoff is None and cell_cutoff is None:
            raise HybridSolverError(
                "Please specify a cutoff, either through lca_cutoff or cell_cutoff"
            )

        super().__init__(prior_transformation)

        self.top_solver = top_solver
        self.bottom_solver = bottom_solver

        # enforce the prior transformations are the same across solvers
        self.top_solver.prior_transformation = prior_transformation
        self.bottom_solver.prior_transformation = prior_transformation

        self.lca_cutoff = lca_cutoff
        self.cell_cutoff = cell_cutoff

        self.threads = threads

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ):
        """The general hybrid solver routine.

        The hybrid solver proceeds by clustering together cells using the
        algorithm stored in the top_solver until a criteria is reached. Once
        this criteria is reached, the bottom_solver is applied to each
        subproblem left over from the "greedy" clustering.

        Args:
            cassiopeia_tree: CassiopeiaTree that stores the character matrix
                and priors for reconstruction.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
            logfile: Location to log progress.
        """
        node_name_generator = solver_utilities.node_name_generator()

        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        unique_character_matrix = character_matrix.drop_duplicates()

        weights = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )
        
        tree = nx.DiGraph()
        # call top-down solver until a desired cutoff is reached.
        _, subproblems, tree = self.apply_top_solver(
            unique_character_matrix,
            list(unique_character_matrix.index),
            tree,
            node_name_generator,
            weights=weights,
            missing_state_indicator=cassiopeia_tree.missing_state_indicator,
        )

        # multi-threaded bottom solver approach
        with multiprocessing.Pool(processes=self.threads) as pool:

            results = list(
                tqdm(
                    pool.starmap(
                        self.apply_bottom_solver,
                        [
                            (
                                cassiopeia_tree,
                                subproblem[0],
                                subproblem[1],
                                logfile,
                                layer,
                            )
                            for subproblem in subproblems
                        ],
                    ),
                    total=len(subproblems),
                )
            )

        for result in results:

            subproblem_tree, subproblem_root = result[0], result[1]

            # check that the only overlapping name is the root, else
            # add a new name so that we don't get edges across the tree
            existing_nodes = [n for n in tree]

            mapping = {}
            for n in subproblem_tree:
                if n in existing_nodes and n != subproblem_root:
                    mapping[n] = next(node_name_generator)

            subproblem_tree = nx.relabel_nodes(subproblem_tree, mapping)

            tree = nx.compose(tree, subproblem_tree)

        # append sample names to the solution and populate the tree
        samples_tree = self.__add_duplicates_to_tree_and_remove_spurious_leaves(tree, character_matrix, node_name_generator)

        leaves = [n for n in samples_tree if samples_tree.out_degree(n) == 0]

        cassiopeia_tree.populate_tree(samples_tree, layer=layer)
        cassiopeia_tree.collapse_unifurcations()

        # collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

    def apply_top_solver(
        self,
        character_matrix: pd.DataFrame,
        samples: List[str],
        tree: nx.DiGraph,
        node_name_generator: Generator[str, None, None],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
        root: Optional[int] = None,
    ) -> Tuple[int, List[Tuple[int, List[str]]]]:
        """Applies the top solver to samples.

        A helper method for applying the top solver to the samples
        until a criteria is hit. Subproblems and the root ID are returned.

        Args:
            character_matrix: Character matrix
            samples: Samples in the subclade of interest.
            tree: In progress tree for the HybridSolver.
            node_name_generator: Generator for creating unique node names
                while applying the top-solver.
            weights: Weights of character-state combinations, derived from
                priors if these are available.
            missing_state_indicator: Indicator for missing data
            root: Node ID of the root in the subtree containing the samples.

        Returns:
            The ID of the node serving as the root of the tree containing the
                samples, and a list of subproblems in the form
                [subtree-root, subtree-samples].
        """

        clades = list(
            self.top_solver.perform_split(
                character_matrix, samples, weights, missing_state_indicator
            )
        )

        root = next(node_name_generator)

        tree.add_node(root)
        if len(clades) == 1:
            for clade in clades[0]:
                tree.add_edge(root, clade)
            return root, [], tree

        new_clades = []
        subproblems = []
        for clade in clades:

            if len(clade) == 0:
                continue

            if self.assess_cutoff(
                clade, character_matrix, missing_state_indicator
            ):
                subproblems += [(root, clade)]
                continue

            new_clades.append(clade)

        for clade in new_clades:
            child, new_subproblems, tree = self.apply_top_solver(
                character_matrix,
                clade,
                tree,
                node_name_generator,
                weights,
                missing_state_indicator,
                root,
            )
            tree.add_edge(root, child)

            subproblems += new_subproblems

        return root, subproblems, tree

    def apply_bottom_solver(
        self,
        cassiopeia_tree: CassiopeiaTree,
        root: int,
        samples=List[str],
        logfile: str = "stdout.log",
        layer: Optional[str] = None,
    ) -> Tuple[nx.DiGraph, int]:
        """Apply the bottom solver to subproblems.

        A private method for solving subproblems identified by the top-down
        solver with the more precise bottom solver for this instantation of
        the HybridSolver. This function will create a unique log file, based
        on the root, set up a new instance of the bottom solver and solve the
        subproblem.

        The function will return a tree for the subproblem and the identifier
        of the root of the tree.

        Args:
            cassiopeia_tree: CassiopeiaTree for the entire dataset. This
                will be subsetted with respect to the samples specified.
            root: Identifier of the root in the master tree
            samples: A list of samples for which to infer a tree.
            logfile: Base location for logging output. A specific logfile
                will be created from this base logfile name.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.

        Returns:
            A tree in the form of a Networkx graph and the original root
                identifier

        """
        if len(samples) == 1:
            subproblem_tree = nx.DiGraph()
            subproblem_tree.add_edge(root, samples[0])
            return subproblem_tree, root

        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        subproblem_character_matrix = character_matrix.loc[samples]

        subtree_root = data_utilities.get_lca_characters(
            subproblem_character_matrix.loc[samples].values.tolist(),
            cassiopeia_tree.missing_state_indicator,
        )

        base_logfile = logfile.split(".log")[0]
        subtree_root_string = "-".join([str(s) for s in subtree_root])
        logfile = f"{base_logfile}_{subtree_root_string}.log"

        subtree = CassiopeiaTree(
            subproblem_character_matrix,
            missing_state_indicator=cassiopeia_tree.missing_state_indicator,
            priors=cassiopeia_tree.priors,
        )
        self.bottom_solver.solve(subtree, logfile=logfile)

        subproblem_tree = subtree.get_tree_topology()
        subproblem_root = [
            n for n in subproblem_tree if subproblem_tree.in_degree(n) == 0
        ][0]
        subproblem_tree.add_edge(root, subproblem_root)

        return subproblem_tree, root

    def assess_cutoff(
        self,
        samples: List[str],
        character_matrix: pd.DataFrame,
        missing_state_indicator: int = -1,
    ) -> bool:
        """Assesses samples with respect to hybrid cutoff.

        Args:
            samples: A list of samples in a clade.
            character_matrix: Character matrix
            missing_state_indicator: Indicator for missing data.

        Returns:
            True if the cutoff is reached, False if not.
        """

        if self.cell_cutoff is None:
            root_states = data_utilities.get_lca_characters(
                character_matrix.loc[samples].values.tolist(),
                missing_state_indicator,
            )

            lca_distances = [
                dissimilarity_functions.hamming_distance(
                    np.array(root_states), character_matrix.loc[u].values
                )
                for u in samples
            ]

            if np.max(lca_distances) <= self.lca_cutoff:
                return True

        else:
            if len(samples) <= self.cell_cutoff:
                return True

        return False

    def __add_duplicates_to_tree_and_remove_spurious_leaves(
        self, tree: nx.DiGraph, character_matrix: pd.DataFrame, node_name_generator: Generator[str, None, None],
    ) -> nx.DiGraph:
        """Append duplicates and prune spurious extant lineages from the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations. If any extant
        nodes that are not in the original character matrix are present, they
        are removed and their lineages are pruned such that the remaining 
        leaves match the set of samples in the character matrix.

        Args:
            tree: The tree after solving
            character_matrix: Character matrix

        Returns:
            The tree with duplicates added and spurious leaves pruned
        """

        character_matrix.index.name = "index"
        duplicate_groups = (
            character_matrix[character_matrix.duplicated(keep=False) == True]
            .reset_index()
            .groupby(character_matrix.columns.tolist())["index"]
            .agg(["first", tuple])
            .set_index("first")["tuple"]
            .to_dict()
        )

        for i in duplicate_groups:
            new_internal_node = next(node_name_generator)
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_groups[i]:
                tree.add_edge(new_internal_node, duplicate)

        # remove extant lineages that don't correspond to leaves
        to_drop = []
        leaves = [n for n in tree if tree.out_degree(n) == 0]
        for l in leaves:
            if l not in character_matrix.index:
                to_drop.append(l)

                parent = [p for p in tree.predecessors(l)][0]
                while tree.out_degree(parent) < 2:
                    to_drop.append(parent)
                    parent = [p for p in tree.predecessors(parent)][0]
        
        tree.remove_nodes_from(to_drop)

        return tree
