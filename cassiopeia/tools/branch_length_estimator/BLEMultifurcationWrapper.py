import copy
from queue import PriorityQueue
from cassiopeia.data import CassiopeiaTree
import networkx as nx
from .BranchLengthEstimator import (
    BranchLengthEstimator,
    BranchLengthEstimatorError,
)


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

    def estimate_branch_lengths(self, tree: CassiopeiaTree):
        binary_topology = binarize_topology(tree.get_tree_topology())
        # For debugging:
        # print(f"binary_topology = {binary_topology.__dict__}")
        tree_binary = CassiopeiaTree(
            character_matrix=tree.get_current_character_matrix(),
            tree=binary_topology,
        )
        tree_binary.reconstruct_ancestral_characters(zero_the_root=True)
        self.__ble_model.estimate_branch_lengths(tree_binary)
        # Copy the times from the binary tree onto the original tree
        times = dict([(v, tree_binary.get_time(v)) for v in tree.nodes])
        tree.set_times(times)


def binarize_topology(tree: nx.DiGraph) -> nx.DiGraph:
    r"""
    Given a tree represented by a networkx DiGraph, it resolves
    multifurcations. The tree is NOT modified in-place.
    The root is made to have only one children, as in a real-life tumor
    (the founding cell never divides immediately!)
    """
    tree = copy.deepcopy(tree)
    node_names = set([n for n in tree])
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    subtree_sizes = {}
    _dfs_subtree_sizes(tree, subtree_sizes, root)
    assert len(subtree_sizes) == len([n for n in tree])

    # First make the root have degree 1.
    if tree.out_degree(root) >= 2:
        children = list(tree.successors(root))
        assert len(children) == tree.out_degree(root)
        # First remove the edges from the root
        tree.remove_edges_from([(root, child) for child in children])
        # Now create the intermediate node and add edges back
        root_child = f"{root}-child"
        if root_child in node_names:
            raise BranchLengthEstimatorError("Node name already exists!")
        tree.add_edge(root, root_child)
        tree.add_edges_from([(root_child, child) for child in children])

    def _dfs_resolve_multifurcations(tree, v):
        children = list(tree.successors(v))
        if len(children) >= 3:
            # Must resolve the multifurcation
            _resolve_multifurcation(tree, v, subtree_sizes, node_names)
        for child in children:
            _dfs_resolve_multifurcations(tree, child)

    _dfs_resolve_multifurcations(tree, root)
    # Check that the tree is binary
    if not (len(tree.nodes) == len(tree.edges) + 1):
        raise BranchLengthEstimatorError("Failed to binarize tree")
    return tree


def _resolve_multifurcation(tree, v, subtree_sizes, node_names):
    r"""
    node_names is used to make sure we don't create a node name that already
    exists.
    """
    children = list(tree.successors(v))
    n_children = len(children)
    assert n_children >= 3

    # Remove all edges from v to its children
    tree.remove_edges_from([(v, child) for child in children])

    # Create the new binary structure
    queue = PriorityQueue()
    for child in children:
        queue.put((subtree_sizes[child], child))

    for i in range(n_children - 2):
        # Coalesce two smallest subtrees
        subtree_1_size, subtree_1_root = queue.get()
        subtree_2_size, subtree_2_root = queue.get()
        assert subtree_1_size <= subtree_2_size
        coalesced_tree_size = subtree_1_size + subtree_2_size + 1
        coalesced_tree_root = f"{v}-coalesce-{i}"
        if coalesced_tree_root in node_names:
            raise BranchLengthEstimatorError("Node name already exists!")
        # For debugging:
        # print(f"Coalescing {subtree_1_root} (sz {subtree_1_size}) and"
        #       f" {subtree_2_root} (sz {subtree_2_size})")
        tree.add_edges_from(
            [
                (coalesced_tree_root, subtree_1_root),
                (coalesced_tree_root, subtree_2_root),
            ]
        )
        queue.put((coalesced_tree_size, coalesced_tree_root))
    # Hang the two subtrees obtained to v
    subtree_1_size, subtree_1_root = queue.get()
    subtree_2_size, subtree_2_root = queue.get()
    assert subtree_1_size <= subtree_2_size
    tree.add_edges_from([(v, subtree_1_root), (v, subtree_2_root)])


def _dfs_subtree_sizes(tree, subtree_sizes, v) -> int:
    res = 1
    for child in tree.successors(v):
        res += _dfs_subtree_sizes(tree, subtree_sizes, child)
    subtree_sizes[v] = res
    return res
