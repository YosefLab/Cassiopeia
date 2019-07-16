import pickle as pic
import networkx as nx

from Cassiopeia.TreeSolver.data_pipeline import convert_network_to_newick_format
from Cassiopeia.TreeSolver.post_process_tree import post_process_tree
import random

import copy

class Cassiopeia_Tree:
	"""
	An abstract class for all nodes in a tree. Unless created manually, these nodes are created in Cassiopeia.TreeSolver.lineage_solver.solver_utils in the `node_parent`
	function. If the node parent already exists (tested by checking for equality with respect to the character states and process id) then we do not create a new node.

	Attributes:
		- name: name of node (this will either be some internal identifier or the cellBC)
		- char_vec: the array of character states, ordered by character.
		- char_string: a string representation of the char_vec, delimited by '|'. Used for quick comparisons between node character states.
		- pid: process id (useful for disambiguating between identical character states traversed on different parts of the tree)
		- is_target: boolean value indicating whether or not these nodes are targets or not.

	Methods:
		- get_character_string: utility function for getting character string
		- get_name: utility for getting the name of the node
		- get_character_vec: utility for getting the character vector
		- get_edit_distance: calculate the edit distance between two nodes
	
	"""

	def __init__(self, method, name = None, network = None, newick = None, character_matrix = None):

		assert network is not None or newick is not None

		assert method in ['greedy', 'hybrid', 'ilp', 'cassiopeia', 'camin-sokal', 'neighbor-joining', 'simulated']

		self.name = name
		self.method = method
		self.network = network
		self.newick = newick
		self.cm = character_matrix

	def dump_network(self, output_name):

		if not self.network:
			self.network = newick_to_network(self.newick)

		pic.dump(self.network, open(output_name, "wb"))

	def dump_newick(self, output_name):

		if not self.newick:
			self.newick = convert_network_to_newick_format(self.network)

		with open(output_name, "w") as f:
			f.write(self.newick)

	def get_network(self):

		if not self.network:
			self.network = newick_to_network(self.newick)

		return self.network

	def get_newick(self):

		if not self.newick:
			self.newick = convert_network_to_newick_format(self.network)

		return self.newick 

	def get_targets(self):

		if not self.network:
			self.network = newick_to_network(self.newick)

		return [n for n in self.network if n.is_target]

	def collapse_edges(self):


		def _collapse(graph, edges_to_collapse):
		
			new_network = nx.DiGraph()
			root = [n for n in graph if graph.in_degree(n) == 0][0]
			for n in nx.dfs_postorder_nodes(graph, source=root):
				if n == root:
					continue
				edge = (list(graph.predecessors(n))[0], n)
				if edge[0].get_character_vec() == edge[1].get_character_vec():
					for p in graph.predecessors(edge[0]):
						new_network.add_edge(p, edge[1])
				else:
					new_network.add_edge(edge[0], edge[1])

			return new_network

		def find_edges_to_collapse(graph):
			edges = []
			source = [n for n in graph if graph.in_degree(n) == 0][0]

			for e in nx.dfs_edges(graph, source=source):
				if e[0].get_character_vec() == e[1].get_character_vec():
					edges.append(e)
			return edges[::-1]

		net = self.network
		root = [n for n in net if net.in_degree(n) == 0][0]

		edges_to_collapse = find_edges_to_collapse(net)
		while len(edges_to_collapse) > 0:
			net = _collapse(net)
			edges_to_collapse = find_edges_to_collapse(net)

		to_remove = []
		for n in net.nodes:
			if net.in_degree(n) == 0 and n != root:
				to_remove.append(n)

		net.remove_nodes_from(to_remove)

		self.network = net
		self.newick = convert_network_to_newick_format(self.network)


	def post_process(self, cm = None):

		if cm is not None:
			self.cm = cm

		assert self.cm is not None

		net = self.get_network().copy()
		copy_dict = {}
		for n in net:
			copy_dict[n] = copy.copy(n)

		net = nx.relabel_nodes(net, copy_dict)

		G = post_process_tree(net, self.cm.copy(), self.method)

		return Cassiopeia_Tree(self.method, network=G)

	def score_parsimony(self, cm = None):

		if cm is not None:
			self.cm = cm

		assert self.cm is not None

		net = self.get_network().copy()
		copy_dict = {}
		for n in net:
			copy_dict[n] = copy.copy(n)

		net = nx.relabel_nodes(net, copy_dict)

		#net = fill_in_tree(net, cm)
		#net = tree_collapse(net)

		root = [n for n in net if net.in_degree(n) == 0][0]

		score = 0
		for e in nx.dfs_edges(net, source=root):
			score += e[0].get_edit_distance(e[1])

		return score

	def generate_triplet(self, targets = None):

		if targets is None:
		
			targets = get_leaves_of_tree(self.network)

		a = random.choice(targets)
		target_nodes_original_network_copy = list(targets)
		target_nodes_original_network_copy.remove(a)
		b = random.choice(target_nodes_original_network_copy)
		target_nodes_original_network_copy.remove(b)
		c = random.choice(target_nodes_original_network_copy)

		return [a, b, c]

	def find_triplet_structure(self, triplet):

		a, b, c = None, None, None

		for n in self.network.nodes:
			if n.char_string == triplet[0].char_string:
				a = n
			if n.char_string == triplet[1].char_string:
				b = n
			if n.char_string == triplet[2].char_string:
				c = n

		a_ancestors = [node for node in nx.ancestors(self.network, a)]
		b_ancestors = [node for node in nx.ancestors(self.network, b)]
		c_ancestors = [node for node in nx.ancestors(self.network, c)]
		ab_common = len(set(a_ancestors) & set(b_ancestors))
		ac_common = len(set(a_ancestors) & set(c_ancestors))
		bc_common = len(set(b_ancestors) & set(c_ancestors))
		index = min(ab_common, bc_common, ac_common)

		true_common = '-'
		if ab_common > bc_common and ab_common > ac_common:
			true_common = 'ab'
		elif ac_common > bc_common and ac_common > ab_common:
			true_common = 'ac'
		elif bc_common > ab_common and bc_common > ac_common:
			true_common = 'bc'

		return true_common, index

	def get_leaves(self):
		"""
		Given a tree, returns all leaf nodes with/without their identifiers

		:param tree:
			networkx tree generated from the simulation process
		:param clip_identifier:
			Boolean flag to remove _identifier from node names
		:return:
			List of leaves of the corresponding tree in string format
		"""

		tree = self.network

		source = [x for x in tree.nodes() if tree.in_degree(x)==0][0]

		max_depth = max(nx.shortest_path_length(tree,source,node) for node in tree.nodes())
		shortest_paths = nx.shortest_path_length(tree,source)

		return [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x) == 1 and shortest_paths[x] == max_depth]



