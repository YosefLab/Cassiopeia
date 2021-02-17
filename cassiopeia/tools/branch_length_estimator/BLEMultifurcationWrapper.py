import copy
from cassiopeia.data import CassiopeiaTree, resolve_multifurcations_networkx
from .BranchLengthEstimator import BranchLengthEstimator


# https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
# permalink: https://stackoverflow.com/a/1445289
class BLEMultifurcationWrapper(BranchLengthEstimator):
    r"""
    Wraps a BranchLengthEstimator.
    When estimating branch lengths:
    1) the tree topology is first copied out
    2) then multifurcations in the tree topology are resolved into a binary
        structure,
    3) then branch lengths are estimated on this binary topology
    4) finally, the node ages are copied back onto the original tree.
    Maximum Parsimony will be used to reconstruct the ancestral states.
    """

    def __init__(self, ble_model: BranchLengthEstimator):
        ble_model = copy.deepcopy(ble_model)
        self.__class__ = type(
            ble_model.__class__.__name__,
            (self.__class__, ble_model.__class__),
            {},
        )
        self.__dict__ = ble_model.__dict__
        self.__ble_model = ble_model

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        binary_topology = resolve_multifurcations_networkx(
            tree.get_tree_topology())
        # For debugging:
        print(f"binary_topology = {binary_topology.__dict__}")
        tree_binary = CassiopeiaTree(
            character_matrix=tree.get_current_character_matrix(),
            tree=binary_topology,
        )
        tree_binary.reconstruct_ancestral_characters(zero_the_root=True)
        self.__ble_model.estimate_branch_lengths(tree_binary)
        # Copy the times from the binary tree onto the original tree
        times = dict([(v, tree_binary.get_time(v)) for v in tree.nodes])
        tree.set_times(times)
