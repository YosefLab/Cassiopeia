import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree

from .FitnessEstimator import FitnessEstimator


class LocalBranchingIndex(FitnessEstimator):
    """
    Local Branching Index (LBI) from Neher et al. 2014
    """
    def __init__(self, tau: float = 1.0):
        self._tau = tau

    def estimate_fitness(self, tree: CassiopeiaTree):
        """
        Fitness estimated as the LBI.
        """
        tau = self._tau
        down = {}
        up = {}
        for v in tree.depth_first_traverse_nodes():
            res = 0
            for u in tree.children(v):
                res += down[u]
            if v != tree.root:
                l = tree.get_branch_length(tree.parent(v), v)
                res = res * np.exp(-l/tau) + tau * (1.0 - np.exp(-l/tau))
            down[v] = res
        for v in tree.depth_first_traverse_nodes(postorder=False):
            if tree.is_root(v):
                up[v] = 0
            else:
                res = 0
                p = tree.parent(v)
                l = tree.get_branch_length(p, v)
                for u in tree.children(p):
                    if u != v:
                        res += np.exp(-l/tau) * down[u]
                res += tau * (1.0 - np.exp(-l/tau))
                res += np.exp(-l/tau) * up[p]
                up[v] = res
        for v in tree.nodes:
            fitness = up[v] + sum([down[u] for u in tree.children(v)])
            tree.set_attribute(v, "fitness", fitness)
