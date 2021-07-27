from cassiopeia.data import CassiopeiaTree
from .BranchLengthEstimator import BranchLengthEstimator
from copy import deepcopy
import numpy as np
from typing import List


class BLEEnsemble(BranchLengthEstimator):
    r"""
    Ensemble several BranchLengthEstimator. Just averages the edge lengths.

    Args:
        branch_length_estimators: List of BranchLengthEstimator to ensemble.
    """

    def __init__(
        self,
        branch_length_estimators: List[BranchLengthEstimator],
    ):
        self.branch_length_estimators = branch_length_estimators

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        branch_length_estimators = self.branch_length_estimators
        trees = []
        for ble in branch_length_estimators:
            tree_i = deepcopy(tree)
            ble.estimate_branch_lengths(tree_i)
            trees.append(tree_i)
        times = dict(
            [
                (
                    node,
                    np.mean([
                        tree_i.get_time(node)
                        for tree_i in trees
                    ])
                )
             for node in tree.nodes
            ]
        )
        tree.set_times(times)
