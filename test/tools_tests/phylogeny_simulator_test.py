from cassiopeia.tools.phylogeny_simulator import generate_perfect_binary_tree,\
    generate_perfect_binary_tree_with_root_branch


def test_generate_perfect_binary_tree_with_fixed_lengths():
    T = generate_perfect_binary_tree(
        generation_branch_lengths=[2, 3]
    )
    newick = T.to_newick_tree_format(print_internal_nodes=True)
    assert(newick == '((3:3,4:3)1:2,(5:3,6:3)2:2)0);')


def test_generate_perfect_binary_tree_with_fixed_lengths_with_root_branch():
    T = generate_perfect_binary_tree_with_root_branch(
        generation_branch_lengths=[2, 3, 4],
    )
    newick = T.to_newick_tree_format(print_internal_nodes=True)
    assert(newick == '(((4:4,5:4)2:3,(6:4,7:4)3:3)1:2)0);')
