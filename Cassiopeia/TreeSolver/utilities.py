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

def tree_collapse2(tree, source):

	collapsing = True

	to_remove = []
	while collapsing:
		newNodes = []

		collapsing = False

		for c in tree.successors(source):

			if c.char_string == source.char_string and tree.out_degree(c) > 0:

				newNodes += [c2 for c2 in tree.successors(c)]
				to_remove.append(c)

				collapsing = True

			else:
				newNodes.append(c)

	tree.remove_nodes_from(to_remove)
	tree.add_edges_from([(source, n) for n in newNodes])

	for c in tree.successors(source):
		tree_collapse2(tree, c)



def tree_collapse2(tree): 

	def collapse_decision(phy): 

		_leaves = [n for n in phy if phy.out_degree(n) == 0]

		for l in _leaves:
			for a in nx.ancestors(phy, l):
				if a.char_string == l.char_string:
					return True

		return False

	while collapse_decision(tree):

		_leaves = [n for n in tree if tree.out_degree(n) == 0]
		to_remove = []
		edges_to_add = []
		for l in _leaves:

			for a in nx.ancestors(tree, l):
				if a.char_string == l.char_string:

					if a in to_remove:
						continue

					to_remove.append(a)

					desc_of_a = tree.successors(a)
					anc_of_a = nx.ancestors(tree, a)


					edges_to_add += [(l, d) for d in desc_of_a if d != l]
					edges_to_add += [(a2, l) for a2 in anc_of_a]

		tree.remove_nodes_from(to_remove)
		tree.add_edges_from(edges_to_add)

		tree.remove_edges_from(tree.selfloop_edges())

	return tree


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
