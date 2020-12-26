import numpy as np
from .tree import Tree


def lineage_tracing_simulator(
    T: Tree,
    mutation_rate: float,
    num_characters: float
) -> None:
    r"""
    Populates the phylogenetic tree T with lineage tracing characters.
    """
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
                    # The character has already mutated; there in nothing to do
                    child_state += node_state[i]
                    continue
                else:
                    # Determine if the character will mutate.
                    mutates =\
                        np.random.exponential(1.0 / mutation_rate) < edge_length
                    if mutates:
                        child_state += '1'
                    else:
                        child_state += '0'
            T.set_state(child, child_state)
            dfs(child, T)
    root = T.root()
    T.set_state(root, '0' * num_characters)
    dfs(root, T)
