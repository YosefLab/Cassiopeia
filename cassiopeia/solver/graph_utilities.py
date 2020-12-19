import itertools
import numpy as np
import networkx as nx
import pandas as pd

from cassiopeia.solver import solver_utilities
from typing import Callable, Dict, List, Optional, Union


def check_if_cut(
    u: Union[int, str], v: Union[int, str], cut: List[Union[int, str]]
) -> bool:
    """Checks if two nodes are on opposite sides of a graph partition.

    Args:
        u: The first node
        v: The second node
        cut: A set of nodes that represents one of the sides of a partition
            on the graph

    Returns:
        Whether nodes u and v are on the same side of the partition
    """
    return ((u in cut) and (not v in cut)) or ((v in cut) and (not u in cut))


def construct_connectivity_graph(
    cm: pd.DataFrame,
    mutation_frequencies: Dict[int, Dict[int, int]],
    missing_char: int,
    samples: List[Union[int, str]],
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> nx.Graph:
    """
    TODO: Optimize with numba?
    Generates connectivity graph for max-cut algorithm.

    Instantiates a graph with a node for each sample. This graph represents a
    supertree over trees generated for each character. For each pair of nodes
    (samples), the edge weight between those nodes is the total number
    of triplets that seperate the nodes using a single character minus the
    total number of triplets where a single character groups them as an ingroup.
    Effectively, the construction of the graph incentivizes the max-cut
    algorithm to group samples with shared mutations together and split samples
    with distant mutations.

    Args:
        cm: The character matrix of observed character states for all samples
        mutation_frequencies: A dictionary containing the frequencies of
            each character/state pair that appear in the character matrix
            restricted to the sample set
        missing_char: The character representing missing values
        samples: A list of samples to build the graph over
        w: A set of optional weights for edges in the connectivity graph

    Returns:
        A connectivity graph constructed over the sample set
    """
    G = nx.Graph()
    for i in samples:
        G.add_node(i)
    for i, j in itertools.combinations(samples, 2):
        # compute similarity scores
        score = 0
        for l in range(cm.shape[1]):
            x = cm.loc[i, :][l]
            y = cm.loc[j, :][l]
            if (x != missing_char and y != missing_char) and (x != 0 or y != 0):
                if w is not None:
                    if x == y:
                        score -= (
                            3
                            * w[l][x]
                            * (
                                len(samples)
                                - mutation_frequencies[l][x]
                                - mutation_frequencies[l][missing_char]
                            )
                        )
                    elif x == 0:
                        score += w[l][y] * (mutation_frequencies[l][y] - 1)
                    elif y == 0:
                        score += w[l][x] * (mutation_frequencies[l][x] - 1)
                    else:
                        score += w[l][x] * (mutation_frequencies[l][x] - 1) + w[
                            l
                        ][y] * (mutation_frequencies[l][y] - 1)
                else:
                    if x == y:
                        score -= 3 * (
                            len(samples)
                            - mutation_frequencies[l][x]
                            - mutation_frequencies[l][missing_char]
                        )
                    elif x == 0:
                        score += mutation_frequencies[l][y] - 1
                    elif y == 0:
                        score += mutation_frequencies[l][x] - 1
                    else:
                        score += (
                            mutation_frequencies[l][x]
                            + mutation_frequencies[l][y]
                            - 2
                        )
        if score != 0:
            G.add_edge(i, j, weight=score)
    return G


def max_cut_improve_cut(
    G: nx.Graph, cut: List[Union[int, str]]
) -> List[Union[int, str]]:
    """A greedy hill-climbing procedure to optimize a partition for the max-cut.

    The procedure is initialized by calculating the improvement to the max-cut
    criterion gained by moving each node across the partition. This improvement
    is defined to be the change in the weight of cut edges when moving this
    node across the cut. The procedure then iteratively chooses the node with
    the best improvement, updating the improvement of its neighbors at each
    step, terminating when the improvement of all nodes is negative or a max
    iteration number is reached.

    Args:
        G: A graph to find an optimized parition over
        cut: A list of nodes that represents one of the sides of a partition
            on the graph

    Returns:
        A new partition that is a local maximum to the max-cut criterion
    """
    ip = {}
    new_cut = cut.copy()
    for i in G.nodes():
        improvement_potential = 0
        for j in G.neighbors(i):
            if check_if_cut(i, j, new_cut):
                improvement_potential -= G[i][j]["weight"]
            else:
                improvement_potential += G[i][j]["weight"]
        ip[i] = improvement_potential
    all_neg = False
    iters = 0
    while (not all_neg) and (iters < 2 * len(G.nodes)):
        best_potential = 0
        best_index = 0
        for i in G.nodes():
            if ip[i] > best_potential:
                best_potential = ip[i]
                best_index = i
        if best_potential > 0:
            for j in G.neighbors(best_index):
                if check_if_cut(best_index, j, new_cut):
                    ip[j] += 2 * G[best_index][j]["weight"]
                else:
                    ip[j] -= 2 * G[best_index][j]["weight"]
            ip[best_index] = -ip[best_index]
            if best_index in new_cut:
                new_cut.remove(best_index)
            else:
                new_cut.append(best_index)
        else:
            all_neg = True
        iters += 1

    return new_cut


def construct_similarity_graph(
    cm: pd.DataFrame,
    mutation_frequencies: Dict[int, Dict[int, int]],
    missing_char: int,
    samples: List[Union[int, str]],
    similarity_function: Callable[
        [List[int], List[int], int, Optional[Dict[int, Dict[int, float]]]],
        float,
    ],
    threshold: int = 0,
    w: Optional[Dict[int, Dict[int, float]]] = None,
) -> nx.Graph:
    """
    TODO: Optimize with numba?
    Generates a similarity graph on the sample set.

    Generates a similarity graph with the samples in the sample set as nodes.
    For each pair of nodes, an edge is created with weight equal to the
    similarity between those samples. Similarity is defined here as the
    number of (non-missing) shared character/state mutations in the character
    vectors of two samples. A minimum similarity threshold is used to exclude
    mutations that are shared by all samples and to impose sparsity on the
    graph, but can be treated as a hyperparameter.

    Args:
        cm: The character matrix of observed character states for all samples
        mutation_frequencies: A dictionary containing the frequencies of
            each character/state pair that appear in the character matrix
            restricted to the sample set
        missing_char: The character representing missing values
        samples: A list of samples to build the graph over
        similarity_function: A function that calculates a similarity score
            between two given samples and their observed mutations
        threshold: A minimum similarity threshold
        w: A set of optional weights for edges in the similarity graph

    Returns:
        A similarity graph constructed over the sample set
    """
    G = nx.Graph()
    for i in samples:
        G.add_node(i)

    for i, j in itertools.combinations(samples, 2):
        s = similarity_function(
            list(cm.loc[i, :]), list(cm.loc[j, :]), missing_char, w
        )
        if s > threshold:
            G.add_edge(i, j, weight=s)

    if len(G.edges) > 1:
        min_edge = min(G.edges(data=True), key=lambda x: x[2]["weight"])
        min_edge_weight = min_edge[2]["weight"]
        G.remove_edge(min_edge[0], min_edge[1])

        to_remove = []
        for node1, node2, d in G.edges(data=True):
            d["weight"] -= min_edge_weight
            if d["weight"] <= 0:
                to_remove.append((node1, node2))

        for edge in to_remove:
            G.remove_edge(edge[0], edge[1])

    return G


def spectral_improve_cut(
    G: nx.Graph, cut: List[Union[int, str]]
) -> List[Union[int, str]]:
    """A greedy hill-climbing procedure minimizing a modified normalized cut.

    The procedure minimizes a partition on a graph for the following objective
    function: weight of edges across cut/ min(weight of edges within each side
    of cut). This is a modified version of the normalized cut. The procedure is
    initialized by calculating the improvement to the objective gained by moving
    each node across the partition. The procedure then iteratively chooses the
    node with the best improvement, updating the improvement of its neighbors at
    each step, terminating when the improvement of all nodes is negative or a
    max iteration number is reached. Essentially, the procedure tries to improve
    a partition by grouping samples with similar mutations together.

    Args:
        G: A graph to find an optimized parition over
        cut: A list of nodes that represents one of the sides of a partition
            on the graph

    Returns:
        A new partition that is a local minimum to the objective function
    """

    def set_improvement_potential(node: Union[int, str]):
        """A helper function to calculate the change to the cut weight by
        moving the node to the other side of the partition.

        Args:
            node: The index of a sample and its respective node in the graph G

        Returns:
            None. Annotates the improvement_potentials dictionary
        """
        # If moving a node across the cut would result in one side having 0
        # weight, that move is disallowed
        if np.isclose(
            min(
                weight_within_side + delta_denominator[node],
                total_weight - weight_within_side - delta_denominator[node],
            ),
            0,
        ):
            improvement_potentials[node] = np.inf
        else:
            # The improvement potential is the change to the weight ratio in
            # moving the node across the cut.
            improvement_potentials[node] = (
                numerator + delta_numerator[node]
            ) / min(
                weight_within_side + delta_denominator[node],
                total_weight - weight_within_side - delta_denominator[node],
            ) - numerator / min(
                weight_within_side, total_weight - weight_within_side
            )

    delta_numerator = {}
    delta_denominator = {}
    improvement_potentials = {}
    new_cut = cut
    total_weight = 2 * sum([G[e[0]][e[1]]["weight"] for e in G.edges()])
    numerator = sum(
        [
            G[e[0]][e[1]]["weight"]
            for e in G.edges()
            if check_if_cut(e[0], e[1], new_cut)
        ]
    )
    weight_within_side = sum(
        [sum([G[u][v]["weight"] for v in G.neighbors(u)]) for u in new_cut]
    )
    if numerator == 0:
        return new_cut

    for node1 in G.nodes:
        # Annotate each node with the change to the weight across the cut and
        # the weight within a side of the cut
        neighbor_weight = sum(
            [G[node1][node2]["weight"] for node2 in G.neighbors(node1)]
        )
        cut_weight = sum(
            [
                G[node1][node2]["weight"]
                for node2 in G.neighbors(node1)
                if check_if_cut(node1, node2, new_cut)
            ]
        )
        delta_numerator[node1] = neighbor_weight - 2 * cut_weight
        if node1 in new_cut:
            delta_denominator[node1] = -neighbor_weight
        else:
            delta_denominator[node1] = neighbor_weight
        # Set the improvement potential for each node
        set_improvement_potential(node1)

    all_pos = False
    iters = 0

    while (not all_pos) and (iters < len(G.nodes)):
        # Negative potentials improve the cut
        best_potential = min(improvement_potentials.values())
        if best_potential < 0:
            best_index = min(
                improvement_potentials, key=improvement_potentials.get
            )
            numerator += delta_numerator[best_index]
            weight_within_side += delta_denominator[best_index]
            for i in G.neighbors(best_index):
                if check_if_cut(best_index, i, new_cut):
                    delta_numerator[i] += 2 * G[best_index][i]["weight"]
                else:
                    delta_numerator[i] -= 2 * G[best_index][i]["weight"]
                set_improvement_potential(i)
            delta_numerator[best_index] = -delta_numerator[best_index]
            delta_denominator[best_index] = -delta_denominator[best_index]
            set_improvement_potential(best_index)
            if best_index in new_cut:
                new_cut.remove(best_index)
            else:
                new_cut.append(best_index)
        else:
            all_pos = True
        iters += 1

    return new_cut
