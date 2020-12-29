from cassiopeia.tools import PerfectBinaryTree, PerfectBinaryTreeWithRootBranch


def test_PerfectBinaryTree():
    tree = PerfectBinaryTree(generation_branch_lengths=[2, 3])\
        .simulate_lineage()
    newick = tree.to_newick_tree_format(print_internal_nodes=True)
    assert(newick == '((3:3,4:3)1:2,(5:3,6:3)2:2)0);')


def test_PerfectBinaryTreeWithRootBranch():
    tree = PerfectBinaryTreeWithRootBranch(generation_branch_lengths=[2, 3, 4])\
        .simulate_lineage()
    newick = tree.to_newick_tree_format(print_internal_nodes=True)
    assert(newick == '(((4:4,5:4)2:3,(6:4,7:4)3:3)1:2)0);')
