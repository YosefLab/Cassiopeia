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

    Also, the probability that a tree with only one internal node is obtained is
    e^-lam * (1 - e^-lam) where lam is the birth rate, so we also check this.
    """
    np.random.seed(1)
    birth_rate = 0.6
    intensities = []
    repetitions = 10000
    topology_hits = 0
    for _ in range(repetitions):
        tree_true = BirthProcess(
            birth_rate=birth_rate, tree_depth=1.0
        ).simulate_lineage()
        if len(tree_true.nodes()) == 4:
            topology_hits += 1
        leaf = np.random.choice(tree_true.leaves())
        n_leaves = len(tree_true.leaves())
        n_hits = tree_true.num_ancestors(leaf) - 1
        intensity = n_leaves / 2 ** n_hits * n_hits
        intensities.append(intensity)
    # Check that the probability of the topology matches
    empirical_topology_prob = topology_hits / repetitions
    theoretical_topology_prob = np.exp(-birth_rate) * (
        1.0 - np.exp(-birth_rate)
    )
    assert np.abs(empirical_topology_prob - theoretical_topology_prob) < 0.02
    inferred_birth_rate = np.array(intensities).mean()
    print(f"{birth_rate} == {inferred_birth_rate}")
    assert np.abs(birth_rate - inferred_birth_rate) < 0.05
