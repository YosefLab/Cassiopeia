from collections import defaultdict
import networkx as nx
import random

from tqdm import tqdm
from Cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from Cassiopeia.TreeSolver.Node import Node

def tree_collapse(tree):
	"""
	Given a networkx graph in the form of a tree, collapse two nodes together if there are no mutations seperating the two nodes

	:param graph: Networkx Graph as a tree
	:return: Collapsed tree as a Networkx object
	"""
	if isinstance(tree, Cassiopeia_Tree):
		graph = tree.get_network()
	else:
		graph = tree

	new = {}
	for node in graph.nodes():

		if isinstance(node, Node):
			new[node] = node.char_string
		else:
			new[node] = node.split("_")[0]

	new_graph = nx.relabel_nodes(graph, new)

	dct = defaultdict(str)
	while len(dct) != len(new_graph.nodes()):
		for node in new_graph:
			if '|' in node:
				dct[node] = node
			else:
				succ = list(new_graph.successors(node))
				if len(succ) == 1:
						if '|' in succ[0]:
							 dct[node] = succ[0]
						elif '|' in dct[succ[0]]:
							dct[node] = dct[succ[0]]
				else:
					if '|' in succ[0] and '|' in succ[1]:
							dct[node] = node_parent(succ[0], succ[1])
					elif '|' in dct[succ[0]] and '|' in succ[1]:
							dct[node] = node_parent(dct[succ[0]], succ[1])
					elif '|' in succ[0] and '|' in dct[succ[1]]:
							dct[node] = node_parent(succ[0], dct[succ[1]])
					elif '|' in dct[succ[0]] and '|' in dct[succ[1]]:
							dct[node] = node_parent(dct[succ[0]], dct[succ[1]])

	new_graph = nx.relabel_nodes(new_graph, dct)
	new_graph.remove_edges_from(new_graph.selfloop_edges())

	final_dct = {}
	for n in new_graph:

		final_dct[n] = Node('state-node', character_vec = n.split("|"))
	
	new_graph = nx.relabel_nodes(new_graph, final_dct)		

	# new2 = {}
	# for node in new_graph.nodes():
	# 	new2[node] = node+'_x'
	# new_graph = nx.relabel_nodes(new_graph, new2)

	return new_graph

def find_parent(node_list):
	parent = []
	if isinstance(node_list[0], Node):
		node_list = [n.get_character_string() for n in node_list]


	num_char = len(node_list[0].split("|"))
	for x in range(num_char):
		inherited = True

		state = node_list[0].split("|")[x]
		for n in node_list:
			if n.split("|")[x] != state:
				inherited = False
			
		if not inherited:
			parent.append("0")
			
		else: 
			parent.append(state)
	
	return Node('state-node', parent, is_target=False)

def fill_in_tree(tree, cm = None):
	
	# rename samples to character strings
	rndct = {}
	for n in tree.nodes:
		if cm is None:
			if '|' in n:
				rndct[n] = Node('state-node', n.split("_")[0].split('|'), is_target = False)
		else:
			if n in cm.index.values:
				rndct[n] = Node('state-node', [str(k) for k in cm.loc[n].values], is_target = True)
			else:
				rndct[n] = Node('state-node', '', is_target=False)

	tree = nx.relabel_nodes(tree, rndct)
	
	root = [n for n in tree if tree.in_degree(n) == 0][0]
	
	# run dfs and reconstruct states 
	anc_dct = {}
	for n in tqdm(nx.dfs_postorder_nodes(tree, root)):
		if '|' not in n.char_string or len(n.char_string) == 0:
			children = list(tree[n].keys())
			for c in range(len(children)):
				if children[c] in anc_dct:
					children[c] = anc_dct[children[c]]
			anc_dct[n] = find_parent(children)
			
	tree = nx.relabel_nodes(tree, anc_dct)

	tree.remove_edges_from(tree.selfloop_edges())

	return tree
