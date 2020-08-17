import pickle as pic
import networkx as nx

from cassiopeia.TreeSolver.utilities import convert_network_to_newick_format
from cassiopeia.TreeSolver.post_process_tree import post_process_tree
from cassiopeia.TreeSolver.simulation_tools.simulation_utils import get_leaves_of_tree
import random
import numpy as np

import copy

from tqdm import tqdm
import hashlib

class Cassiopeia_Tree:
	"""
	An abstract class for trees used in Cassiopeia. Basically a wrapper for a networkx file and a newick string that can be used to interface
	with other applications. 

	Attributes:
		- method: the algorithm used to reconstruct this tree.
		- name: name of the tree
		- network: a networkx object representing the tree
		- newick: the newick string corresponding to the tree.
		- cm: the character matrix used as input to the tree solver. 
		- alternative_solutions: a list of networkx objects, each of which corresponds to an alternative solution

	Methods:
		- dump_network: write out networkx object to a .pkl file
		- dump_newick: write out the newick file as text
		- get_network: retrieve the networkx object
		- get_newick: retrieve the newick text as a string
		- get_targets: get target nodes in tree
		- post_process: post-process the tree by adding cells with same allele onto tree; also make depth uniform and remove non-sample leaves.
		- score_parsimony: get the parsimony of the tree.
		- generate_triplet: obtain a random triplet from the tree.
		- find_triplet_structure: find the mrca of the triplet (used to score triplets)
		- get_leaves: get the leaves of the tree.
	
	"""

	def __init__(self, method, name = None, network = None, newick = None, character_matrix = None, alternative_solutions = None, base_network = None):
		"""
		Initialize the Cassiopeia_Tree object.

		:param method:
			Algorithm used to reconstruct the tree
		:param name:
			Name of the tree.
		:param network:
			Networkx object representing the tree.
		:param newick:
			Newick string for tree.
		:param character_matrix:
			character matrix used as input for reconstructing the tree.
		:param alternative_solutions:
			a list of alternative solutions (each of which is a networkx object)

		:return: 
			None
		"""

		assert network is not None or newick is not None

		assert method in ['greedy', 'greedy_heritable', 'hybrid', 'ilp', 'cassiopeia', 'camin-sokal', 'neighbor-joining', 'simulated', 'from_newick']

		self.name = name
		self.method = method
		self.network = network
		self.newick = newick
		self.cm = character_matrix
		self.alternative_solutions = alternative_solutions
		self.base_network = base_network

	def dump_network(self, output_name):
		"""
		Write network as a pickle file.

		:param output_name:
			File path to write to.
		:return:
			None
		"""

		if not self.network:
			self.network = newick_to_network(self.newick)

		pic.dump(self.network, open(output_name, "wb"))

	def dump_newick(self, output_name):
		"""
		Write newick string to text.

		:param output_name:
			File path to write to.
		:return:
			None
		"""

		if not self.newick:
			self.newick = convert_network_to_newick_format(self.network)

		with open(output_name, "w") as f:
			f.write(self.newick)

	def get_network(self):
		"""
		Get networkx object. 

		:return:
			Networkx object. 
		"""

		if not self.network:
			self.network = newick_to_network(self.newick)

		return self.network

	def get_newick(self):
		"""
		Get newick string. 

		:return:
			Newick string.
		"""

		if not self.newick:
			self.newick = convert_network_to_newick_format(self.network)

		return self.newick 

	def get_targets(self):
		"""
		Get targets of tree (as determined by Node.is_target boolean).

		:return:
			List of target Nodes. 
		"""

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
		"""
		Post process the tree by:
			- adding in non-unique samples as leaves
			- pruning off leaves that are not targets
			- Adding 'dummy' edges to make the depth of the tree uniform.

		:return:
			A Cassiopeia_Tree that is post-processed. 
		"""

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
		"""
		Score the parsimony of the tree.
	
		:param cm:
			Character matrix, if the Cassiopeia_Tree object does not already have one stored.

		:return:
			An integer representing the number of mutations in the tree.
		"""

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
			score += e[0].get_mut_length(e[1])

		return score

	def score_likelihood(self, priors):

		net = self.network
		root = [n for n in net if net.in_degree(n) == 0][0]

		score = 0
		for e in nx.dfs_edges(net, source=root):
			score += e[0].get_mut_length(e[1], priors=priors)

		return score

	def generate_triplet(self, targets = None):
		"""
		Generate a random triplet of targets in the tree. 

		:param targets:
			Targets to choose from. If this is None, select randomly from the leaves of the tree (these should be
			targets themselves).
		
		:return: 
			A list of Nodes corresponding to the triplet. 
		"""

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
		"""
		Find the structure of the triplet -- i.e. are A and B more closely related to one another or B and C?

		:param triplet:
			A list of Nodes (length = 3)
		
		:return: 
			The structure of the triplet and the minimum number of ancestors that overlap between each Node in the triplet.
		"""

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

		return [n for n in tree if tree.out_degree(n) == 0 and tree.in_degree(n) == 1] 
		
		# source = [x for x in tree.nodes() if tree.in_degree(x)==0][0]
		
		# max_depth = max(nx.shortest_path_length(tree,source,node) for node in tree.nodes())
		# shortest_paths = nx.shortest_path_length(tree,source)

		# return [x for x in tree.nodes() if tree.out_degree(x)==0 and tree.in_degree(x) == 1 and shortest_paths[x] == max_depth]

	def sample_alternative_solutions(self, maximum_alt_solutions = 100):

		alt_solutions = []

		target_charstrings = [(n.char_string, n.pid) for n in self.network if n.is_target]

		num_solutions = np.prod([len(self.alternative_solutions[r]) for r in self.alternative_solutions.keys()])

		num_considered_solutions = 0
		sol_identifiers = []  # keep track of solutions already sampled

		# we'll sample maximum_alt_solutions from the set of possible solutions
		pbar = tqdm(
			total=maximum_alt_solutions, desc="Enumerating alternative solutions"
		)

		while num_considered_solutions < min(num_solutions, maximum_alt_solutions):

			current_sol = []
			for r in self.alternative_solutions.keys():
				res_list = self.alternative_solutions[r]
				current_sol.append(np.random.choice(len(res_list)))

			if tuple(current_sol) not in sol_identifiers:
				
				new_network = self.base_network.copy()

				for i, r in zip(current_sol, self.alternative_solutions.keys()):

					sub_net = self.alternative_solutions[r][i]

					# reattach alternative solution
					r_parent = None
					r_parents = [p for p in self.base_network.predecessors(r)]
					if len(r_parents) > 0:
						r_parent = r_parents[0]
					
					rn = [n for n in sub_net if sub_net.in_degree(n) == 0][0]
					
					new_network.add_edges_from(sub_net.edges(data=True))
					new_network.add_nodes_from(sub_net.nodes(data=True))

					if r_parent != None:
						new_network.add_edge(r_parent, rn)
				
				# assign targets
				seen_charstrings = [n.char_string for n in self.base_network if n.is_target]
				for n in new_network:
					
					if (n.char_string, n.pid) in target_charstrings and n.char_string not in seen_charstrings:
						n.is_target = True
						seen_charstrings.append(n.char_string)

				alt_solutions.append(new_network)

				sol_identifiers.append(tuple(current_sol))
				num_considered_solutions += 1

				pbar.update(1)  # update progress bar

		return alt_solutions
		


