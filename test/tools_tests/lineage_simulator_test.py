import numpy as np

from cassiopeia.tools import (
    PerfectBinaryTree,
    PerfectBinaryTreeWithRootBranch,
    BirthProcess,
)


def test_PerfectBinaryTree():
    tree = PerfectBinaryTree(
        generation_branch_lengths=[2, 3]
    ).simulate_lineage()
    newick = tree.to_newick_tree_format(print_internal_nodes=True)
    assert newick == "((3:3,4:3)1:2,(5:3,6:3)2:2)0);"


def test_PerfectBinaryTreeWithRootBranch():
    tree = PerfectBinaryTreeWithRootBranch(
        generation_branch_lengths=[2, 3, 4]
    ).simulate_lineage()
    newick = tree.to_newick_tree_format(print_internal_nodes=True)
    assert newick == "(((4:4,5:4)2:3,(6:4,7:4)3:3)1:2)0);"


def test_BirthProcess():
    r"""
    Generate tree, then choose a random lineage can count how many nodes are on
    the lineage. This is the number of times the process triggered on that
    lineage.
    """
    np.random.seed(1)
    birth_rate = 0.6
    intensities = []
    for _ in range(10000):
        tree_true = BirthProcess(
            birth_rate=birth_rate, tree_depth=1.0
        ).simulate_lineage()
        leaf = np.random.choice(tree_true.leaves())
        n_leaves = len(tree_true.leaves())
        n_hits = tree_true.num_ancestors(leaf) - 1
        intensity = n_leaves / 2 ** n_hits * n_hits
        intensities.append(intensity)
    inferred_birth_rate = np.array(intensities).mean()
    print(f"{birth_rate} == {inferred_birth_rate}")
    assert np.abs(birth_rate - inferred_birth_rate) < 0.05
