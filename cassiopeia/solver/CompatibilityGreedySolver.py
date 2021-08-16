"""
This file stores a subclass of the CassiopeiaGreedy solver that uses a
compatibility-based heuristic to find character-splits.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
from networkx.readwrite.json_graph import jit
import numpy as np
import pandas as pd

from cassiopeia.solver import (
    GreedySolver,
    missing_data_methods,
    solver_utilities,
)


class CompatibilityGreedySolver(GreedySolver.GreedySolver):
    """Compatibility-based Greedy Solver.

    Uses a compatibility-based heuristic to split cells during a
    CassiopeiaGreedy routine.

    Args:
        prior_transformation: A function defining a transformation on the priors
            in forming weights to scale frequencies and the contribution of
            each mutation in the connectivity graph. One of the following:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        prior_transformation: Function to transform priors, if these are
            available.
        missing_data_classifier: Function to classify missing data during
            character splits.
    """

    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(prior_transformation)
        self.missing_data_classifier = missing_data_classifier

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Performs a partition using a compatibility-based heuristic.

        Splits cells using a heuristic measuring compatibility of
        (character, state) pairs.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """

        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )

        mutation_frequencies = self.compute_mutation_frequencies(
            samples, character_matrix, missing_state_indicator
        )

        compatibility_graph = self.create_compatibility_graph(
            character_matrix.values,
            weights=weights,
            missing_state_indicator=missing_state_indicator,
        )
       
        best_score = -np.inf
        chosen_character = 0
        chosen_state = 0
        for character in mutation_frequencies:
            for state in mutation_frequencies[character]:

                frequency = mutation_frequencies[character][state]

                if state != missing_state_indicator and state != 0:

                    # Avoid splitting on mutations shared by all samples
                    if (
                        frequency
                        < len(samples)
                        - mutation_frequencies[character][
                            missing_state_indicator
                        ]
                    ):

                        score = frequency + self.get_risk(character, state, compatibility_graph) 
                        if best_score < score:
                            chosen_character, chosen_state = character, state
                            best_score = score
                        
        if chosen_state == 0:
            return samples, []

        left_set = []
        right_set = []
        missing = []

        unique_character_array = character_matrix.to_numpy()
        sample_names = list(character_matrix.index)

        for i in sample_indices:
            if unique_character_array[i, chosen_character] == chosen_state:
                left_set.append(sample_names[i])
            elif (
                unique_character_array[i, chosen_character]
                == missing_state_indicator
            ):
                missing.append(sample_names[i])
            else:
                right_set.append(sample_names[i])

        left_set, right_set = self.missing_data_classifier(
            character_matrix,
            missing_state_indicator,
            left_set,
            right_set,
            missing,
            weights=weights,
        )

        return left_set, right_set

    def create_compatibility_graph(
        self,
        character_matrix: np.array,
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> nx.DiGraph:
        """Create a compatibility graph.

        Creates a compatibility graph for checking the lower bound of the greedy
        split risk. In this graph, each node is a (character, state) pair and
        each edge between (character, state) pairs indicates incompatibility.
        These edges are directed, and are weighted by the number of samples that
        must be removed in the source to obtain compatibility. If prior-weights
        are specified, then this number is multipled by the prior-weight of each
        (character, state) pair.

        Args:
            character_matrix: Character matrix of samples.
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A directed compatibility graph for the CompatibilityGreedySolver. 
        """

        compatibility_graph = nx.DiGraph()
        for i in range(character_matrix.shape[1]):

            for s1 in np.unique(character_matrix[:, i]):

                if s1 == missing_state_indicator or s1 == 0:
                    continue

                row_i = np.where(character_matrix[:, i] == s1)[0]

                for j in range(character_matrix.shape[1]):

                    if i == j:
                        continue

                    for s2 in np.unique(character_matrix[:, j]):

                        if s2 == missing_state_indicator or s2 == 0:
                            continue

                        row_j = np.where(character_matrix[:, j] == s2)[0]

                        if (
                            len(np.intersect1d(row_i, row_j)) > 0
                            and len(np.setdiff1d(row_j, row_i))
                        ) == 0:
                            if weights:
                                w = weights[j][s2] * len(row_j)
                            else:
                                w = len(row_j)
                                
                            compatibility_graph.add_edge(
                                f"{i}-{s1}", f"{j}-{s2}", weight=w
                            )

        return compatibility_graph

    def get_risk(
        self, character: int, state: int, compatibility_graph: nx.DiGraph
    ) -> float:
        """Score the risk of a node in the compatibility graph.

        Given a node in the directed compatibility graph, the risk is the
        weight of all edges emanating from the node. Qualitatively, this risk
        captures the amount of incompatibility that the node introduces.

        Args:
            character: Character index
            state: State indicator
            compatibility_graph: The directed compatibility graph.

        Returns:
            The risk of the node. 
        """
        risk = 0
        for e in compatibility_graph.out_edges(f"{character}-{state}"):
            risk += compatibility_graph[e[0]][e[1]]["weight"]
        return risk