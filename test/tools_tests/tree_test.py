import networkx as nx

from cassiopeia.tools.tree import Tree


def test_to_newick_tree_format():
    r"""
    Example tree based off https://itol.embl.de/help.cgi#upload .
    The most basic newick example should give:
    (2:0.5,(4:0.3,5:0.4):0.2):0.1);
    """
    T = nx.DiGraph()
    T.add_nodes_from([0, 1, 2, 3, 4, 5])
    T.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    T = Tree(T)
    T.set_edge_lengths(
        [(0, 1, 0.1),
         (1, 2, 0.5),
         (1, 3, 0.2),
         (3, 4, 0.3),
         (3, 5, 0.4)]
    )
    T.set_states(
        [(0, '0000000000'),
         (1, '1000000000'),
         (2, '1111000000'),
         (3, '1110000000'),
         (4, '1110000111'),
         (5, '1110111111')]
    )
    res = T.to_newick_tree_format(print_internal_nodes=False)
    assert(res == "((2:0.5,(4:0.3,5:0.4):0.2):0.1));")
    res = T.to_newick_tree_format(
        print_node_names=False,
        print_internal_nodes=True,
        append_state_to_node_name=True)
    assert(res == "((_1111000000:0.5,(_1110000111:0.3,_1110111111:0.4)"
                  "_1110000000:0.2)_1000000000:0.1)_0000000000);")
    res = T.to_newick_tree_format(print_internal_nodes=True)
    assert(res == "((2:0.5,(4:0.3,5:0.4)3:0.2)1:0.1)0);")
    res = T.to_newick_tree_format(print_node_names=False)
    assert(res == "((:0.5,(:0.3,:0.4):0.2):0.1));")
    res = T.to_newick_tree_format(
        print_internal_nodes=True,
        add_N_to_node_id=True)
    assert(res == "((N2:0.5,(N4:0.3,N5:0.4)N3:0.2)N1:0.1)N0);")
    res = T.to_newick_tree_format(
        print_internal_nodes=True,
        append_state_to_node_name=True,
        add_N_to_node_id=True)
    assert(res == "((N2_1111000000:0.5,(N4_1110000111:0.3,N5_1110111111:0.4)"
                  "N3_1110000000:0.2)N1_1000000000:0.1)N0_0000000000);")
    res = T.to_newick_tree_format(
        print_internal_nodes=True,
        print_pct_of_mutated_characters_along_edge=True,
        add_N_to_node_id=True)
    assert(res == "((N2:0.5[&&NHX:muts=0.33],(N4:0.3[&&NHX:muts=0.43],"
           "N5:0.4[&&NHX:muts=0.86])N3:0.2[&&NHX:muts=0.22])"
           "N1:0.1[&&NHX:muts=0.10])N0);")


def test_reconstruct_ancestral_states():
    T = nx.DiGraph()
    T.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    T.add_edges_from([(10, 11),
                      (11, 13),
                      (13, 0), (13, 1),
                      (11, 14),
                      (14, 2), (14, 3),
                      (10, 12),
                      (12, 15),
                      (15, 4), (15, 5),
                      (12, 16),
                      (16, 6), (16, 7), (16, 8), (16, 9)])
    T = Tree(T)
    T.set_states(
        [(0, '01101110100'),
         (1, '01211111111'),
         (2, '01322121111'),
         (3, '01432122111'),
         (4, '01541232111'),
         (5, '01651233111'),
         (6, '01763243111'),
         (7, '01873240111'),
         (8, '01983240111'),
         (9, '01093240010'),
         ]
    )
    T.reconstruct_ancestral_states()
    assert(T.get_state(10) == '00000000000')
    assert(T.get_state(11) == '01000100100')
    assert(T.get_state(13) == '01001110100')
    assert(T.get_state(14) == '01002120111')
    assert(T.get_state(12) == '01000200010')
    assert(T.get_state(15) == '01001230111')
    assert(T.get_state(16) == '01003240010')


def test_reconstruct_ancestral_states_DREAM_challenge_tree_25():
    T = nx.DiGraph()
    T.add_nodes_from(list(range(21)))
    T.add_edges_from([(9, 8), (8, 10), (8, 7), (7, 11), (7, 12), (9, 6),
                      (6, 2), (2, 0), (0, 13), (0, 14), (2, 1), (1, 15),
                      (1, 16), (6, 5), (5, 3), (3, 17), (3, 18), (5, 4),
                      (4, 19), (4, 20)])
    T = Tree(T)
    T.set_states(
        [(10, '0022100000'),
         (11, '0022100000'),
         (12, '0022100000'),
         (13, '2012000220'),
         (14, '2012000200'),
         (15, '2012000100'),
         (16, '2012000100'),
         (17, '0001110220'),
         (18, '0001110220'),
         (19, '0000210220'),
         (20, '0000210220'),
         ]
    )
    T.reconstruct_ancestral_states()
    assert(T.get_state(7) == '0022100000')
    assert(T.get_state(8) == '0022100000')
    assert(T.get_state(0) == '2012000200')
    assert(T.get_state(1) == '2012000100')
    assert(T.get_state(2) == '2012000000')
    assert(T.get_state(3) == '0001110220')
    assert(T.get_state(4) == '0000210220')
    assert(T.get_state(5) == '0000010220')
    assert(T.get_state(6) == '0000000000')
    assert(T.get_state(9) == '0000000000')
