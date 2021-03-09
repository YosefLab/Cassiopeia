import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LineageTracingDataSimulator import (
    LineageTracingDataSimulator,
)


class IIDExponentialLineageTracer(LineageTracingDataSimulator):
    r"""
    Characters evolve IID over the lineage, with the same given mutation rate.

    Args:
        mutation_rate: The mutation rate of each character (same for all).
        num_characters: The number of characters.
    """

    def __init__(self, mutation_rate: float, num_characters: float):
        self.mutation_rate = mutation_rate
        self.num_characters = num_characters

    def overlay_data(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        num_characters = self.num_characters
        mutation_rate = self.mutation_rate
        states = {}

        def dfs(node: str, tree: CassiopeiaTree):
            node_state = states[node]
            for child in tree.children(node):
                # Compute the state of the child
                child_state = []
                edge_length = tree.get_branch_length(node, child)
                # print(f"{node} -> {child}, length {edge_length}")
                assert edge_length >= 0
                for i in range(num_characters):
                    # See what happens to character i
                    if node_state[i] != 0:
                        # The character has already mutated; there in nothing
                        # to do
                        child_state += [node_state[i]]
                        continue
                    else:
                        # Determine if the character will mutate.
                        mutates = (
                            np.random.exponential(1.0 / mutation_rate)
                            < edge_length
                        )
                        if mutates:
                            child_state += [1]
                        else:
                            child_state += [0]
                states[child] = child_state
                dfs(child, tree)

        root = tree.root
        states[root] = [0] * num_characters
        dfs(root, tree)
        tree.initialize_all_character_states(states)
