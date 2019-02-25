import networkx as nx

def node_to_string(sample):
	"""
	Given a sample in list format [['0','1','0','2'], uniq_ident] converts to string format, 'Ch1|Ch2|....|Chn_Identifier'

	:param sample:
		Sample in format [['0','1','0','2'], uniq_ident]
	:return:
		String formatted version of a given sample
	"""
	return "|".join(sample[0]) + "_" + str(sample[1])

def get_leaves_of_tree(tree, clip_identifier=False):
	"""
	Given a tree, returns all leaf nodes with/without their identifiers

	:param tree:
		networkx tree generated from the simulation process
	:param clip_identifier:
		Boolean flag to remove _identifier from node names
	:return:
		List of leaves of the corresponding tree in string format
	"""
	source = [x for x in tree.nodes() if tree.in_degree(x)==0][0]

	max_depth = max(nx.shortest_path_length(tree,source,node) for node in tree.nodes())
	shortest_paths = nx.shortest_path_length(tree,source)

	if clip_identifier:
		return [x[:x.index('_')] for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x)==1 and shortest_paths[x] == max_depth]

	else:

		return [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x) == 1 and shortest_paths[x] == max_depth]
