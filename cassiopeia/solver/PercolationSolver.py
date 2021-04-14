from collections import defaultdict
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class PercolationSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    The PercolationSolver implements a top-down algorithm that recursively
    partitions the sample set based on similarity in the observed mutations.
    It is an implicit version of Aho's algorithm for tree discovery (1981).
    At each recursive step, the similarities of each sample pair are embedded
    as edges in a graph with weight equal to the similarity between the nodes.
    The graph is then percolated by removing the minimum edges until multiple
    connected components are produced. Ultimately, this groups clusters of
    samples that share strong similarity amongst themselves. If more than two
    connected components are produced by the percolation procedure, then the
    components are clustered by applying the specified solver to the LCAs of
    the clusters, obeying parsimony

    TODO(richardyz98): Experiment to find the best default similarity function

    Args:
        joining_solver: The CassiopeiaSolver that is used to cluster groups of
            samples in the case that the percolation procedure generates more
            than two groups of samples in the partition
        prior_transformation: A function defining a transformation on the priors
            in forming weights to scale the contribution of each mutation in
            the similarity graph
        similarity_function: A function that calculates a similarity score
            between two given samples and their observed mutations. The default
            is "hamming_distance_without_missing"
        threshold: The minimum similarity threshold. A similarity threshold of 1
            for example means that only samples with similarities above 1 will
            have an edge between them in the graph. Acts as a hyperparameter
            that controls the sparsity of the graph by filtering low
            similarities.


    Attributes:
        joining_solver: The CassiopeiaSolver that is used to cluster groups of
            samples after percolation steps that produce more than two groups
        prior_transformation: Function to use when transforming priors into
            weights.
        similarity_function: A function that calculates a similarity score
            between two given samples and their observed mutations
        threshold: A minimum similarity threshold

    """

    def __init__(
        self,
        joining_solver: CassiopeiaSolver.CassiopeiaSolver,
        prior_transformation: str = "negative_log",
        similarity_function: Optional[
            Callable[
                [
                    List[int],
                    List[int],
                    int,
                    Optional[Dict[int, Dict[int, float]]],
                ],
                float,
            ]
        ] = dissimilarity_functions.hamming_similarity_without_missing,
        threshold: Optional[int] = 0,
    ):

        super().__init__(prior_transformation)

        self.joining_solver = joining_solver
        self.threshold = threshold
        self.similarity_function = similarity_function

    def solve(self, cassiopeia_tree: CassiopeiaTree):
        """Implements a solving procedure for the Percolation Algorithm.

        The procedure recursively splits a set of samples to build a tree. At
        each partition of the samples produced by the percolation procedure,
        an ancestral node is created and each side of the partition is placed
        as a daughter clade of that node. This continues until each side of
        the partition is comprised only of single samples. If an algorithm
        cannot produce a split on a set of samples, then those samples are
        placed as sister nodes and the procedure terminates, generating a
        polytomy in the tree. This function will populate a tree inside the
        input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree storing a character matrix and
                priors.
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(
            samples: List[Union[str, int]],
            tree: nx.DiGraph,
            unique_character_matrix: pd.DataFrame,
            priors: Dict[int, Dict[int, float]],
            weights: Dict[int, Dict[int, float]],
            missing_state_indicator: int,
        ):

            if len(samples) == 1:
                return samples[0]
            # Partitions the set of samples by percolating a similarity graph
            clades = list(
                self.percolate(
                    unique_character_matrix,
                    samples,
                    priors,
                    weights,
                    missing_state_indicator,
                )
            )
            # Generates a root for this subtree with a unique int identifier
            root = len(tree.nodes) + 1
            tree.add_node(root)

            for clade in clades:
                if len(clade) == 0:
                    clades.remove(clade)

            # If unable to return a split, generate a polytomy and return
            if len(clades) == 1:
                for clade in clades[0]:
                    tree.add_edge(root, clade)
                return root
            # Recursively generate the subtrees for each daughter clade
            for clade in clades:
                child = _solve(
                    clade,
                    tree,
                    unique_character_matrix,
                    priors,
                    weights,
                    missing_state_indicator,
                )
                tree.add_edge(root, child)
            return root

        weights = None
        priors = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )
            priors = cassiopeia_tree.priors

        # extract character matrix
        character_matrix = cassiopeia_tree.get_current_character_matrix()
        unique_character_matrix = character_matrix.drop_duplicates()

        tree = nx.DiGraph()
        tree.add_nodes_from(list(unique_character_matrix.index))

        _solve(
            list(unique_character_matrix.index),
            tree,
            unique_character_matrix,
            priors,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )

        # Collapse 0-mutation edges and append duplicate samples
        tree = solver_utilities.collapse_tree(
            tree,
            True,
            character_matrix,
            cassiopeia_tree.missing_state_indicator,
        )
        tree = self.__add_duplicates_to_tree(tree, character_matrix)

        cassiopeia_tree.populate_tree(tree)

    def percolate(
        self,
        character_matrix: pd.DataFrame,
        samples: List[str],
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """The function used by the percolation algorithm to partition the
        set of samples in two.

        First, a pairwise similarity graph is generated with samples as nodes
        such that edges between a pair of nodes is some provided function on
        the number of character/state mutations shared. Then, the algorithm
        removes the minimum edge (in the case of ties all are removed) until
        the graph is split into multiple connected components. If there are more
        than two connected components, the procedure joins them until two remain.
        This is done by inferring the mutations of the LCA of each sample set
        obeying Camin-Sokal Parsimony, and then clustering the groups of samples
        based on their LCAs. The provided solver is used to cluster the groups
        into two clusters.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            priors: A dictionary storing the probability of each character
                mutating to a particular state.
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        unique_character_array = character_matrix.to_numpy()

        G = nx.Graph()
        G.add_nodes_from(sample_indices)

        # Add edge weights into the similarity graph
        edge_weight_buckets = defaultdict(list)
        for i, j in itertools.combinations(sample_indices, 2):
            similarity = self.similarity_function(
                unique_character_array[i, :],
                unique_character_array[j, :],
                missing_state_indicator,
                weights,
            )
            if similarity > self.threshold:
                edge_weight_buckets[similarity].append((i, j))
                G.add_edge(i, j)

        if len(G.edges) == 0:
            return samples, []

        connected_components = list(nx.connected_components(G))
        sorted_edge_weights = sorted(edge_weight_buckets, reverse=True)

        # Percolate the similarity graph by continuously removing the minimum
        # edge until at least two components exists
        while len(connected_components) <= 1:
            min_weight = sorted_edge_weights.pop()
            for edge in edge_weight_buckets[min_weight]:
                G.remove_edge(edge[0], edge[1])
            connected_components = list(nx.connected_components(G))

        # If the number of connected components > 2, merge components by
        # joining the most similar LCAs of each component until
        # only 2 remain
        partition_sides = []

        if len(connected_components) > 2:
            for c in range(len(connected_components)):
                connected_components[c] = list(connected_components[c])
            lcas = {}
            component_to_nodes = {}
            # Find the LCA of the nodes in each connected component
            for ind in range(len(connected_components)):
                component_identifier = "component" + str(ind)
                component_to_nodes[component_identifier] = connected_components[
                    ind
                ]
                character_vectors = [
                    list(i)
                    for i in list(
                        unique_character_array[connected_components[ind], :]
                    )
                ]
                lcas[component_identifier] = data_utilities.get_lca_characters(
                    character_vectors, missing_state_indicator
                )
            # Build a tree on the LCA characters to cluster the components
            lca_tree = CassiopeiaTree(
                pd.DataFrame.from_dict(lcas, orient="index"),
                missing_state_indicator=missing_state_indicator,
                priors=priors,
            )

            self.joining_solver.solve(lca_tree)
            grouped_components = []

            # Take the split at the root as the clusters of components
            # in the split, ignoring unifurcations
            current_node = lca_tree.root
            while len(grouped_components) == 0:
                successors = lca_tree.children(current_node)
                if len(successors) == 1:
                    current_node = successors[0]
                else:
                    for i in successors:
                        grouped_components.append(lca_tree.leaves_in_subtree(i))

            # For each component in each cluster, take the nodes in that
            # component to form the final split
            for cluster in grouped_components:
                sample_index_group = []
                for component in cluster:
                    sample_index_group.extend(component_to_nodes[component])
                partition_sides.append(sample_index_group)
        else:
            for c in range(len(connected_components)):
                partition_sides.append(list(connected_components[c]))

        # Convert from component indices back to the sample names in the
        # original character matrix
        sample_names = list(character_matrix.index)
        partition_named = []
        for sample_index_group in partition_sides:
            sample_name_group = []
            for sample_index in sample_index_group:
                sample_name_group.append(sample_names[sample_index])
            partition_named.append(sample_name_group)

        return partition_named

    def __add_duplicates_to_tree(
        self, tree: nx.DiGraph, character_matrix: pd.DataFrame
    ) -> nx.DiGraph:
        """Takes duplicate samples and places them in the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations.

        Args:
            tree: The tree to have duplicates added to
            character_matrix: Character matrix

        Returns:
            A tree with duplicates added
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
            if len(tree.nodes) == 1:
                new_internal_node = len(duplicate_groups[i]) + 1
            else:
                new_internal_node = (
                    max([n for n in tree.nodes if type(n) == int]) + 1
                )
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_groups[i]:
                tree.add_edge(new_internal_node, duplicate)

        return tree
