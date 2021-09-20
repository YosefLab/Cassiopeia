from cassiopeia.data.CassiopeiaTree import CassiopeiaTree

from .FitnessEstimator import FitnessEstimator


class GenerationNumberFitnessEstimator(FitnessEstimator):
    """
    Fitness estimated as the generation of each node, divided by
    its depth.

    For the root, the fitness is the mean fitness of all leaves in the tree
    """
    def estimate_fitness(self, tree: CassiopeiaTree):
        """
        Fitness estimated as the MLE birth rate for each node.
        """
        def dfs(v: str, generation: int):
            if not tree.is_root(v):
                tree.set_attribute(v, "fitness", generation / (tree.get_time(v) - tree.get_time(tree.root)))
            for u in tree.children(v):
                dfs(u, generation+1)
        dfs(tree.root, generation=0)
        tree.set_attribute(tree.root, "fitness", sum([tree.get_attribute(leaf, "fitness") for leaf in tree.leaves]) / len(tree.leaves))
