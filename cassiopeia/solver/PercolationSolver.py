from collections import defaultdict
import itertools
import networkx as nx
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.solver import GreedySolver
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
    until mulitple connected components are produced. The algorithm enforces
    binary partitions if there are more than two connected components using a
    neighbor-joining procedure.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_function: A function defining a transformation on the priors
            in forming weights to scale the contribution of each mutuation in
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
            out
        weights: Weights on character/mutation pairs, derived from priors
        similarity_function: A function that calculates a similarity score
            between two given samples and their observed mutations
        threshold: A minimum similarity threshold
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        prior_function: Optional[Callable[[float], float]] = None,
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
        ] = None,
        threshold: Optional[int] = 0,
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)

        self.threshold = threshold
        if similarity_function:
            self.similarity_function = similarity_function
        else:
            self.similarity_function = similarity_functions.hamming_similarity

    def perform_split(
        self,
        mutation_frequencies: Dict[int, Dict[int, int]],
        samples: List[int] = None,
    ) -> Tuple[List[int], List[int]]:
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
            mutation_frequencies: A dictionary containing the frequencies of
                each character/state pair that appear in the character matrix
                restricted to the sample set
            samples: A list of samples to partition

        Returns:
            A tuple of lists, representing the left and right partitions
        """

        G = nx.Graph()
        for v in samples:
            G.add_node(v)

        # Add edge weights into the similarity graph
        edge_weights_to_pairs = defaultdict(list)
        for i, j in itertools.combinations(samples, 2):
            similarity = self.similarity_function(
                list(self.unique_character_matrix.loc[i, :]),
                list(self.unique_character_matrix.loc[j, :]),
                self.missing_char,
                self.weights,
            )
            if similarity >= self.threshold:
                edge_weights_to_pairs[similarity].append((i, j))
                G.add_edge(i, j)

        if len(G.edges) == 0:
            return samples, []

        connected_components = list(nx.connected_components(G))
        sorted_edge_weights = sorted(edge_weights_to_pairs, reverse=True)

        # Percolate the similarity graph by continuously removing the minimum
        # edge if only 1 component exists
        while len(connected_components) <= 1:
            min_weight = sorted_edge_weights.pop()
            for edge in edge_weights_to_pairs[min_weight]:
                G.remove_edge(edge[0], edge[1])
            connected_components = list(nx.connected_components(G))
        for i in range(len(connected_components)):
            connected_components[i] = list(connected_components[i])

        # If the number of connected components > 2, merge components by
        # greedily joining the most similar LCAs of each component until
        # only 2 remain
        if len(connected_components) > 2:
            new_clust_num = len(connected_components)
            lcas = {}
            cluster_membership = {}
            for i in range(len(connected_components)):
                cluster_membership[i] = list(connected_components[i])
                character_vectors = self.unique_character_matrix.loc[
                    connected_components[i], :
                ].values.tolist()
                lcas[i] = solver_utilities.get_lca_characters(
                    character_vectors, self.missing_char
                )
            while len(cluster_membership) > 2:
                best_similarity = 0
                to_merge = []
                for cluster1, cluster2 in itertools.combinations(
                    cluster_membership, 2
                ):
                    similarity = self.similarity_function(
                        lcas[cluster1],
                        lcas[cluster2],
                        self.missing_char,
                        self.weights,
                    )
                    if similarity >= best_similarity:
                        best_similarity = similarity
                        to_merge = [cluster1, cluster2]
                new_lca = solver_utilities.get_lca_characters(
                    [lcas[to_merge[0]], lcas[to_merge[1]]], self.missing_char
                )
                lcas[new_clust_num] = new_lca
                cluster_membership[new_clust_num] = cluster_membership.pop(
                    to_merge[0]
                ) + cluster_membership.pop(to_merge[1])
                new_clust_num += 1

            return list(cluster_membership.values())

        return connected_components
