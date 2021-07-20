from copy import deepcopy
from cassiopeia.data import CassiopeiaTree
from .BranchLengthEstimator import BranchLengthEstimator


class IgnoreCharactersWrapper(BranchLengthEstimator):
    r"""
    All characters are set to missing and the wrapped BranchLengthEstimator
    is run. This leads to a class of naive baselines that ignore the
    character matrix information, i.e. they only rely on the topology.

    Args:
        branch_length_estimator: The wrapper BranchLengthEstimator
    """

    def __init__(
        self,
        branch_length_estimator: BranchLengthEstimator,
    ):
        self.branch_length_estimator = branch_length_estimator

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        tree_original = tree
        tree = deepcopy(tree_original)
        for leaf in tree.leaves:
            tree.set_character_states(leaf, [tree.missing_state_indicator] * tree.n_character)
        tree.reconstruct_ancestral_characters(zero_the_root=True)
        self.branch_length_estimator.estimate_branch_lengths(tree)
        times = dict([(node, tree.get_time(node)) for node in tree.nodes])
        tree_original.set_times(times)
