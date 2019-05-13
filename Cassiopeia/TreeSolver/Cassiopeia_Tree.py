import pickle as pic
import networkx as nx

from Cassiopeia.TreeSolver import convert_network_to_newick_format
from Cassiopeia.TreeSolver.post_process_tree import post_process_tree
from Cassiopeia.TreeSolver.simulation_tools.validation import tree_collapse, fill_in_tree

import copy

class Cassiopeia_Tree:

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

	def get_leaves(self):

		if not self.network:
			self.network = newick_to_network(self.newick)

		return [n for n in self.network if self.network.out_degree(n) == 0]

	def get_targets(self):

		if not self.network:
			self.network = newick_to_network(self.newick)

		return [n for n in self.network if n.is_target]

	def post_process(self, cm = None):

		if cm is not None:
			self.cm = cm

		assert self.cm is not None

		net = self.get_network().copy()
		copy_dict = {}
		for n in net:
			copy_dict[n] = copy.copy(n)

		net = nx.relabel_nodes(net, copy_dict)

		return post_process_tree(net, self.cm.copy(), self.method)

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





