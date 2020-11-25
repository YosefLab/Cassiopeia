import itertools
import networkx as nx
import pandas as pd

from cassiopeia.solver import solver_utilities
from typing import Dict, List, Optional, Union


def check_if_cut(u: int, v: int, cut: List[int]) -> bool:
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
    mutation_frequencies: Dict[int, Dict[str, int]],
    missing_char: str,
    samples: List[int] = None,
    w: Optional[Dict[int, Dict[str, float]]] = None,
) -> nx.DiGraph:
    """Generates connectivity graph for max-cut algorithm.

    Instantiates a graph with a node for each sample. This graph represents a
    supertree over trees generated for each character. For each pair of nodes
    (samples), the edge weight between those nodes is the sum of the number
    of triplets that can be seperated using each character. If the nodes are an
    ingroup on that character (the states at that character are the same), then
    the edge weight is negative, otherwise it is positive. Effectively,
    the construction of the graph incentivizes the max-cut algorithm to group
    samples with shared mutations together and split samples with distant
    mutations.

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
            x = cm.iloc[i, l]
            y = cm.iloc[j, l]
            if (x != missing_char and y != missing_char) and (
                x != "0" or y != "0"
            ):
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
                    elif x == "0":
                        score += w[l][y] * (mutation_frequencies[l][y] - 1)
                    elif y == "0":
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
                    elif x == "0":
                        score += mutation_frequencies[l][y] - 1
                    elif y == "0":
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


def max_cut_improve_cut(G: nx.DiGraph, cut: List[int]):
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
