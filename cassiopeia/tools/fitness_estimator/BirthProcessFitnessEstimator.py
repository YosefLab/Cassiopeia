from typing import Tuple

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree

from .FitnessEstimator import FitnessEstimator


class BirthProcessFitnessEstimator(FitnessEstimator):
    """
    Fitness estimated as the MLE birth rate for each node.

    Assumes that the tree is ultrametric. Does not model subsampling effects.

    Args:
        smooth: If true, the fitness of each node will be smoothed out to be the
            mean fitness of its ancestors (including itself).
    """

    def __init__(
        self, smooth: bool = True, leaf_average_for_internal_nodes: bool = False
    ):
        self._smooth = smooth
        self._leaf_average_for_internal_nodes = leaf_average_for_internal_nodes

    def estimate_fitness(self, tree: CassiopeiaTree):
        """
        Fitness estimated as the MLE birth rate for each node.
        """
        fitness = {}

        def dfs(v: str) -> Tuple[int, float]:
            """
            The number of internal nodes below here,
            and total length of the tree.
            """
            ch = tree.children(v)
            if len(ch) == 0:
                # Is a leaf
                # Use the last branch length seen.
                branch_length = tree.get_branch_length(
                    tree.parent(tree.parent(v)), tree.parent(v)
                )
                fitness[v] = 1.0 / branch_length
                return 0, 0
            else:
                # Is internal
                num_internal = 0
                length = 0
                for u in ch:
                    num_internal_u, length_u = dfs(u)
                    num_internal += num_internal_u
                    if not tree.is_leaf(u):
                        num_internal += 1
                    length += length_u + tree.get_branch_length(v, u)
                if num_internal == 0:
                    # Border case: this is a cherry
                    # Use the last branch length seen
                    branch_length = tree.get_branch_length(tree.parent(v), v)
                    fitness[v] = 1.0 / branch_length
                else:
                    fitness[v] = num_internal / length
                return num_internal, length

        dfs(tree.root)

        # Now smooth out the fitness
        smoothed_fitness = {}
        if self._smooth:

            def dfs(v: str, num_ancestors: int, sum_fitness_ancestors: float):
                num_ancestors += 1
                sum_fitness_ancestors += fitness[v]
                smoothed_fitness[v] = sum_fitness_ancestors / num_ancestors
                for u in tree.children(v):
                    dfs(u, num_ancestors, sum_fitness_ancestors)

            dfs(tree.root, 0, 0)
        else:
            smoothed_fitness = fitness

        final_fitness = {}
        if self._leaf_average_for_internal_nodes:

            def dfs(v: str) -> Tuple[int, float]:
                """
                The sum of fitness of the leaves in this subtree,
                and the size of the subtree
                """
                ch = tree.children(v)
                if len(ch) == 0:
                    # Is a leaf
                    final_fitness[v] = smoothed_fitness[v] / 1
                    return 1, smoothed_fitness[v]
                else:
                    n_leaves = 0
                    sum_of_leaves_fitness = 0
                    for u in ch:
                        n_leaves_u, sum_of_leaves_fitness_u = dfs(u)
                        n_leaves += n_leaves_u
                        sum_of_leaves_fitness += sum_of_leaves_fitness_u
                    final_fitness[v] = sum_of_leaves_fitness / n_leaves
                    return n_leaves, sum_of_leaves_fitness

            dfs(tree.root)
        else:
            final_fitness = smoothed_fitness

        for node in tree.nodes:
            tree.set_attribute(node, "fitness", final_fitness[node])
