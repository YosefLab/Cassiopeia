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

    def __init__(self, smooth: bool = True):
        self._smooth = smooth

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
                # Use the reflection heuristic.
                branch_length = tree.get_branch_length(tree.parent(v), v)
                fitness[v] = 0.5 / branch_length
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
                    # Use the reflection heuristic to create pseudo-births.
                    branch_length = tree.get_branch_length(v, ch[0])
                    fitness[v] = 0.5 / branch_length
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

        for node in tree.nodes:
            tree.set_attribute(node, "fitness", smoothed_fitness[node])
