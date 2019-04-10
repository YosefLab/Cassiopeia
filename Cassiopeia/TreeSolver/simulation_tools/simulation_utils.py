import networkx as nx

from Cassiopeia.TreeSolver import Node, Cassiopeia_Tree

def node_to_string(sample, pid=None):
	"""
	Given a sample in list format [['0','1','0','2'], uniq_ident] converts to a Node with name "node_<uniq>" and a character string 
	of the character list.

	:param sample:
		Sample in format [['0','1','0','2'], uniq_ident]
	:return:
		a new Node.
	"""
	parr = sample[0]
	return Node("node" + str(sample[1]), parr, pid = pid, is_target=False)

def get_leaves_of_tree(tree):
	"""
	Given a tree, returns all leaf nodes with/without their identifiers

	:param tree:
		networkx tree generated from the simulation process
	:param clip_identifier:
		Boolean flag to remove _identifier from node names
	:return:
		List of leaves of the corresponding tree in string format
	"""

	if isinstance(tree, Cassiopeia_Tree):
		tree = tree.get_network()

	source = [x for x in tree.nodes() if tree.in_degree(x)==0][0]

	max_depth = max(nx.shortest_path_length(tree,source,node) for node in tree.nodes())
	shortest_paths = nx.shortest_path_length(tree,source)

	return [x for x in tree.nodes() if tree.out_degree(x)==0 and shortest_paths[x] == max_depth]
