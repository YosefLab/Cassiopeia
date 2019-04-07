import networkx as nx
from Cassiopeia.TreeSolver import Node
import Cassiopeia.TreeSolver.lineage_solver as ls 

import sys
stdout_backup = "testlog"

def test_greedy_simple():

	n1 = Node('a', [1,0,0,0,0])
	n2 = Node('b', [1,0,0,1,0])
	n3 = Node('c', [1,0,0,2,0])
	n4 = Node('d', [1,2,0,1,0])
	n5 = Node('e', [1,1,0,1,0])
	n6 = Node('f', [1,0,3,2,0])
	n7 = Node('g', [0,0,0,0,1])
	n8 = Node('h', [0,1,0,0,1])
	n9 = Node('i', [0,1,2,0,1])
	n10 = Node('j', [0,1,1,0,1])

	nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]

	tree = ls.solve_lineage_instance(nodes, method="greedy")
	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1 

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)


def test_hybrid_simple():

	n1 = Node('a', [1,0,0,0,0])
	n2 = Node('b', [1,0,0,1,0])
	n3 = Node('c', [1,0,0,2,0])
	n4 = Node('d', [1,2,0,1,0])
	n5 = Node('e', [1,1,0,1,0])
	n6 = Node('f', [1,0,3,2,0])
	n7 = Node('g', [0,0,0,0,1])
	n8 = Node('h', [0,1,0,0,1])
	n9 = Node('i', [0,1,2,0,1])
	n10 = Node('j', [0,1,1,0,1])

	nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]

	with open(stdout_backup, "w") as f:
		sys.stdout = f
		tree = ls.solve_lineage_instance(nodes, method="hybrid", hybrid_subset_cutoff=3)

	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1 

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)

def test_ilp_simple():

	n1 = Node('a', [1,0,0,0,0])
	n2 = Node('b', [1,0,0,1,0])
	n3 = Node('c', [1,0,0,2,0])
	n4 = Node('d', [1,2,0,1,0])
	n5 = Node('e', [1,1,0,1,0])
	n6 = Node('f', [1,0,3,2,0])
	n7 = Node('g', [0,0,0,0,1])
	n8 = Node('h', [0,1,0,0,1])
	n9 = Node('i', [0,1,2,0,1])
	n10 = Node('j', [0,1,1,0,1])

	nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]
	with open(stdout_backup, "w") as f:
		sys.stdout = f
		tree = ls.solve_lineage_instance(nodes, method="ilp")

	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1 

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)


def test_greedy_parallel_evo():

	n = Node('a', [1,1,2,0])
	n2 = Node('b', [1,1,3,0])
	n3 = Node('c', [2,1,1,0])
	n4 = Node('d', [2,1,3,0])
	n5 = Node('e', [1,3,1,'-'])
	n6 = Node('f', [1, '-', '-', '1'])
	n7 = Node('g', [1,1,0, 2])

	nodes = [n, n2, n3, n4, n5,n6, n7]

	tree = ls.solve_lineage_instance(nodes, method='greedy')
	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)

	multi_parents = [n for n in net if net.in_degree(n) > 1]

	assert len(multi_parents) == 0

def test_hybrid_parallel_evo():

	n = Node('a', [1,1,2,0])
	n2 = Node('b', [1,1,3,0])
	n3 = Node('c', [2,1,1,0])
	n4 = Node('d', [2,1,3,0])
	n5 = Node('e', [1,3,1,'-'])
	n6 = Node('f', [1, '-', '-', '1'])
	n7 = Node('g', [1,1,0, 2])

	nodes = [n, n2, n3, n4, n5,n6, n7]

	with open(stdout_backup, "w") as f:
		sys.stdout = f
		tree = ls.solve_lineage_instance(nodes, method='hybrid', hybrid_subset_cutoff=2)
	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)

	multi_parents = [n for n in net if net.in_degree(n) > 1]

	assert len(multi_parents) == 0

def test_ilp_parallel_evo():

	n = Node('a', [1,1,2,0])
	n2 = Node('b', [1,1,3,0])
	n3 = Node('c', [2,1,1,0])
	n4 = Node('d', [2,1,3,0])
	n5 = Node('e', [1,3,1,'-'])
	n6 = Node('f', [1, '-', '-', '1'])
	n7 = Node('g', [1,1,0, 2])

	nodes = [n, n2, n3, n4, n5,n6, n7]

	with open(stdout_backup, "w") as f:
		sys.stdout = f
		tree = ls.solve_lineage_instance(nodes, method='ilp')
	net = tree.get_network()

	roots = [n for n in net if net.in_degree(n) == 0]

	assert len(roots) == 1

	root = roots[0]

	targets = [n for n in net if n.is_target]

	assert len(targets) == len(nodes)

	for t in targets:
		assert nx.has_path(net, root, t)

	multi_parents = [n for n in net if net.in_degree(n) > 1]

	assert len(multi_parents) == 0

