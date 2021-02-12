from collections import defaultdict
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import GreedySolver
from cassiopeia.solver import NeighborJoiningSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import similarity_functions
from cassiopeia.solver import solver_utilities


class PercolationSolver(GreedySolver.GreedySolver):
    """
    TODO: Experiment to find the best default similarity function
    The PercolationSolver implements a top-down algorithm that recursively
    partitions the sample set based on similarity in the observed mutations.
    It is an implicit version of Aho's algorithm for tree discovery (1981).
    At each recursive step, the similarities of each sample pair are embedded
    in a graph. The graph is then percolated by removing the minimum edges
    until multiple connected components are produced. The algorithm enforces
    binary partitions if there are more than two connected components using a
    neighbor-joining procedure.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_transformation: A function defining a transformation on the priors
            in forming weights to scale the contribution of each mutation in
            the similarity graph
        similarity_function: A function that calculates a similarity score
            between two given samples and their observed mutations
        threshold: A minimum similarity threshold to include an edge in the
            similarity graph

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        unique_character_matrix: A character matrix with duplicate rows filtered
        duplicate_groups: A mapping of samples to the set of duplicates that
            share the same character vector. Uses the original sample names
        weights: Weights on character/mutation pairs, derived from priors
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
        ] = similarity_functions.hamming_similarity,
        threshold: Optional[int] = 0,
    ):

        super().__init__(prior_transformation)

        self.joining_solver = joining_solver
        self.threshold = threshold
        self.similarity_function = similarity_function

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[str],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """The function used by the percolation algorithm to generate a
        partition of the samples.

        First, a pairwise similarity graph is generated with samples as nodes
        such that edges between a pair of nodes is some provided function on
        the number of character/state mutations shared. Then, the algorithm
        removes the minimum edge (in the case of ties all are removed) until
        the graph is split into multiple connected components. If there are more
        than two connected components, the procedure joins them until two remain.
        This is done by inferring the mutations of the LCA of each sample set
        obeying Camin-Sokal Parsimony, and then performing a neighbor-joining
        procedure on these LCAs using the provided similarity function.

        Args:
            samples: A list of samples, represented by their string names


        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        unique_character_array = character_matrix.to_numpy()

        G = nx.Graph()
        for v in sample_indices:
            G.add_node(v)

        # Add edge weights into the similarity graph
        edge_weight_buckets = defaultdict(list)
        for i, j in itertools.combinations(sample_indices, 2):
            similarity = self.similarity_function(
                unique_character_array[i, :],
                unique_character_array[j, :],
                missing_state_indicator,
                weights,
            )
            if similarity >= self.threshold:
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
            for i in range(len(connected_components)):
                component_to_nodes[i] = connected_components[i]
                character_vectors = [
                    list(i)
                    for i in list(
                        unique_character_array[connected_components[i], :]
                    )
                ]
                lcas[i] = data_utilities.get_lca_characters(
                    character_vectors, missing_state_indicator
                )

            # Build a tree on the LCA characters to cluster the components
            lca_tree = CassiopeiaTree(
                pd.DataFrame.from_dict(lcas, orient="index"),
                missing_state_indicator=missing_state_indicator,
                priors="?",
            )

            self.joining_solver.solve(lca_tree)
            lca_network = lca_tree.get_network()
            grouped_components = []
            root = [
                n for n in lca_network if lca_network.in_degree(n) == 0
            ][0]

            # Take the bifurcation at the root as the two clusters of components
            # in the split, ignoring unifurcations
            current_node = root
            while len(grouped_components) == 0:
                successors = list(lca_network.successors(current_node))
                if len(successors) == 1:
                    current_node = successors[0]
                else:
                    for i in successors:
                        grouped_components.append(
                            solver_utilities.get_leaf_children(
                                lca_network, i
                            )
                        )

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

        # Convert from indices back to the sample names in the original
        # character matrix
        sample_names = list(character_matrix.index)
        partition_named = []
        for sample_index_group in partition_sides:
            sample_name_group = []
            for sample_index in sample_index_group:
                sample_name_group.append(
                    sample_names[sample_index]
                )
            partition_named.append(sample_name_group)

        return partition_named
