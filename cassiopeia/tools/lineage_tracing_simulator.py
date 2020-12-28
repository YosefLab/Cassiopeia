import abc

import numpy as np

from .tree import Tree


class LineageTracingSimulator(abc.ABC):
    r"""
    Abstract base class for all lineage tracing simulators.
    """
    @abc.abstractmethod
    def overlay_lineage_tracing_data(self, tree: Tree) -> None:
        r"""
        Annotates the tree's nodes with lineage tracing character vectors.
        Operates on the tree in-place.
        """


class IIDExponentialLineageTracer():
    r"""
    Characters evolve IID over the lineage, with the same rate.
    """
    def __init__(
        self,
        mutation_rate: float,
        num_characters: float
    ):
        self.mutation_rate = mutation_rate
        self.num_characters = num_characters

    def overlay_lineage_tracing_data(self, T: Tree) -> None:
        r"""
        Populates the phylogenetic tree T with lineage tracing characters.
        """
        num_characters = self.num_characters
        mutation_rate = self.mutation_rate

        def dfs(node: int, T: Tree):
            node_state = T.get_state(node)
            for child in T.children(node):
                # Compute the state of the child
                child_state = ''
                edge_length = T.get_age(node) - T.get_age(child)
                # print(f"{node} -> {child}, length {edge_length}")
                assert(edge_length >= 0)
                for i in range(num_characters):
                    # See what happens to character i
                    if node_state[i] != '0':
                        # The character has already mutated; there in nothing
                        # to do
                        child_state += node_state[i]
                        continue
                    else:
                        # Determine if the character will mutate.
                        mutates =\
                            np.random.exponential(1.0 / mutation_rate)\
                            < edge_length
                        if mutates:
                            child_state += '1'
                        else:
                            child_state += '0'
                T.set_state(child, child_state)
                dfs(child, T)
        root = T.root()
        T.set_state(root, '0' * num_characters)
        dfs(root, T)
