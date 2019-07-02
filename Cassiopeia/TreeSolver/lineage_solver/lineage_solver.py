from __future__ import print_function
import sys

import concurrent.futures
import random
import functools
import multiprocessing
import networkx as nx
import numpy as np
import traceback
import hashlib
from collections import defaultdict

from .greedy_solver import root_finder, greedy_build
from .ILP_solver import generate_mSteiner_model, solve_steiner_instance
from .solver_utils import build_potential_graph_from_base_graph
from Cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from Cassiopeia.TreeSolver.Node import Node

def solve_lineage_instance(_target_nodes, prior_probabilities = None, method='hybrid', threads=8, hybrid_subset_cutoff=200, time_limit=1800, max_neighborhood_size=10000, 
							seed=None, num_iter=-1):
	"""
	Aggregated lineage solving method, which given a set of target nodes, will find the maximum parsimony tree
	accounting the given target nodes

	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param method:
		The method used for solving the problem ['ilp, 'hybrid', 'greedy']
			- ilp: Attempts to solve the problem based on steiner tree on the potential graph
				   (Recommended for instances with several hundred samples at most)
			- greedy: Runs a greedy algorithm to find the maximum parsimony tree based on choosing the most occurring split in a
				   top down fasion (Algorithm scales to any number of samples)
			- hybrid: Runs the greedy algorithm until there are less than hybrid_subset_cutoff samples left in each leaf of the
				   tree, and then returns a series of small instance ilp is then run on these smaller instances, and the
				   resulting graph is created by merging the smaller instances with the greedy top-down tree
	:param threads:
		The number of threads to use in parallel for the hybrid algorithm
	:param hybrid_subset_cutoff:
		The maximum number of nodes allowed before the greedy algorithm terminates for a given leaf node
	:return:
		A reconstructed subgraph representing the nodes
	"""

	target_nodes = [n.get_character_string() + "_" + n.name for n in _target_nodes]

	node_name_dict = dict(zip([n.split("_")[0] for n in target_nodes], [n + "_target" for n in target_nodes]))

	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	# Account for possible cases where the state was not observed in the frequency table, thus we
	# set the value of this prior probability to the minimum probability observed
	character_mutation_mapping = defaultdict(int)
	min_prior = 1
	if prior_probabilities != None:
		for i in prior_probabilities.keys():
			for j in prior_probabilities[i].keys():
				min_prior = min(min_prior, prior_probabilities[i][j])

		for node in target_nodes:
			node_list = node.split("_")[0].split('|')
			for i in range(0, len(node_list)):
				char = node_list[i]
				if char != '0' and char != '-':
					character_mutation_mapping[(str(i), char)] += 1

				for (i,j) in character_mutation_mapping:
					if j not in prior_probabilities[int(i)]:
						prior_probabilities[int(i)][j] = min_prior


	# clip identifier for now, but make sure to add later
	target_nodes = [n.split("_")[0] for n in target_nodes]

	#target_nodes = list(set(target_nodes))
	master_root = root_finder(target_nodes)
	if method == "ilp":


		subgraph, r, pid = find_good_gurobi_subgraph(master_root, target_nodes, node_name_dict, prior_probabilities, time_limit, 1, max_neighborhood_size, seed = seed, num_iter=num_iter)
		clean_ilp_network(subgraph)

		rdict = {}
		for n in subgraph:
			spl = n.split("_")
			nn = Node("state-node", spl[0].split("|"), is_target = False)
			if len(spl) > 1:
				nn.pid = spl[1]
			if spl[0] in node_name_dict:
				nn.is_target = True
			rdict[n] = nn

		state_tree = nx.relabel_nodes(subgraph, rdict)



		return Cassiopeia_Tree(method="ilp", network=state_tree, name="Cassiopeia_state_tree")

	if method == "hybrid":


		network, target_sets = greedy_build(target_nodes, priors=prior_probabilities, cutoff=hybrid_subset_cutoff)

		print("Using " + str(min(multiprocessing.cpu_count(), threads)) + " threads, " + str(multiprocessing.cpu_count()) + " available.", flush=True)
		executor = concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), threads))
		print("Sending off Target Sets: " + str(len(target_sets)), flush=True)

		# just in case you've hit a target node during the greedy reconstruction, append name at this stage
		# so the composition step doesn't get confused when trying to join to the root.
		network = nx.relabel_nodes(network, node_name_dict)

		futures = [executor.submit(find_good_gurobi_subgraph, root, targets, node_name_dict, None, time_limit, 1, max_neighborhood_size, seed, num_iter) for root, targets in target_sets]
		concurrent.futures.wait(futures)
		for future in futures:
			res, r, pid = future.result()
			new_names = {}
			for n in res:
				if res.in_degree(n) == 0 or n == r:
					new_names[n] = n
				else:
					new_names[n] = n + "_" + str(pid)
					res = nx.relabel_nodes(res, new_names)
			network = nx.compose(network, res)

		rdict = {}
		for n in network:
			spl = n.split("_")
			nn = Node("state-node", spl[0].split("|"), is_target = False)

			if len(spl) == 2:
				if "target" in n:
					nn.is_target = True

			if len(spl) > 2:
				nn.is_target = True
				nn.pid = spl[2]

			rdict[n] = nn

		state_tree = nx.relabel_nodes(network, rdict)

		return Cassiopeia_Tree(method="hybrid", network=state_tree, name="Cassiopeia_state_tree")

	if method == "greedy":
		graph = greedy_build(target_nodes, priors=prior_probabilities, cutoff=-1, targets=target_nodes)[0]

		rdict = {}
		for n in graph:
			spl = n.split("_")
			nn = Node('state-node', spl[0].split("|"), is_target = False)
			if len(spl) > 1:
				nn.pid = spl[1] 
			if spl[0] in node_name_dict:
				nn.is_target = True
			rdict[n] = nn

		state_tree = nx.relabel_nodes(graph, rdict)

		return Cassiopeia_Tree(method='greedy', network=state_tree, name='Cassiopeia_state_tree')

	else:
		raise Exception("Please specify one of the following methods: ilp, hybrid, greedy")



def reraise_with_stack(func):

	@functools.wraps(func)
	def wrapped(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except Exception as e:
			traceback_str = traceback.format_exc(e)
			raise StandardError("Error occurred. Original traceback "
								"is\n%s\n" % traceback_str)

	return wrapped

@reraise_with_stack
def find_good_gurobi_subgraph(root, targets, node_name_dict, prior_probabilities, time_limit, num_threads, max_neighborhood_size, seed=None, num_iter=-1):
	"""
	Sub-Function used for multi-threading in hybrid method

	:param root:
		Sub-root of the subgraph that is attempted to be reconstructed
	:param targets:
		List of sub-targets for a given subroot where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param time_limit:
		Length of time allowed for ILP convergence.
	:param num_threads:
		Number of threads to be used during ILP solving.
	:param max_neighborhood_size:
		Maximum size of potential graph allowed.
	:return:
		Optimal ilp subgraph for a given subset of nodes in the time limit allowed.
	"""

	pid = hashlib.md5(root.encode('utf-8')).hexdigest()

	print("Started new thread for: " + str(root) + " (num targets = " + str(len(targets)) + ") , pid = " + str(pid), flush=True)

	if len(set(targets)) == 1:
		graph = nx.DiGraph()
		graph.add_node(node_name_dict[root])
		return graph, root, pid

	potential_network_priors, lca_dist = build_potential_graph_from_base_graph(targets, root, priors=prior_probabilities, max_neighborhood_size=max_neighborhood_size, pid = pid)

	# network was too large to compute, so just run greedy on it
	if potential_network_priors is None:
		subgraph = greedy_build(targets, priors=prior_probabilities, cutoff=-1)[0]
		subgraph = nx.relabel_nodes(subgraph, node_name_dict)
		print("Max Neighborhood Exceeded", flush=True)
		return subgraph, root, pid

	print("Potential Graph built with maximum LCA of " + str(lca_dist) + " (pid: " + str(pid) + "). Proceeding to solver.")

	for l in potential_network_priors.selfloop_edges():
		potential_network_priors.remove_edge(l[0], l[1])

	nodes = list(potential_network_priors.nodes())
	encoder = dict(zip(nodes, list(range(len(nodes)))))
	decoder = dict((v, k) for k, v in encoder.items())

	assert len(encoder) == len(decoder)

	_potential_network = nx.relabel_nodes(potential_network_priors, encoder)
	_targets = map(lambda x: encoder[x], targets)

	model, edge_variables = generate_mSteiner_model(_potential_network, encoder[root], _targets)
	subgraph = solve_steiner_instance(model, _potential_network, edge_variables, MIPGap=.01, detailed_output=False, time_limit=time_limit, num_threads = num_threads, seed=seed, num_iter=num_iter)[0]

	subgraph = nx.relabel_nodes(subgraph, decoder)

	# remove spurious roots left in the solution
	subgraph_roots = [n for n in subgraph if subgraph.in_degree(n) == 0]
	print(subgraph_roots, str(pid), flush=True)
	print(root + " pid: " + str(pid), flush=True)
	for r in subgraph_roots:
		if r != root:
			subgraph.remove_node(r)

	node_name_dict_cleaned = {}
	for n in node_name_dict.keys():
		if n in targets:
			node_name_dict_cleaned[n] = node_name_dict[n]

	subgraph = nx.relabel_nodes(subgraph, node_name_dict_cleaned)
	clean_ilp_network(subgraph)

	r_name = root
	if root in node_name_dict:
		r_name = node_name_dict[root]

	return subgraph, r_name, pid

def clean_ilp_network(network):
	"""
	Post-processes networks after an ILP run. At times the ILP will return Steiner Trees which are not necessarily 
	tres (specifically, a node may have more than one parent). To get around this we remove these spurious edges so
	that the trees returned are truly trees. CAUTION: this will modify the network in place. 

	:param network: 
		Network returned from an ILP run.
	:return:
		None. 

	"""


	for u, v in network.edges():
		if u == v:
			network.remove_edge(u,v)
	trouble_nodes = [node for node in network.nodes() if network.in_degree(node) > 1]
	for node in trouble_nodes:
		pred = network.predecessors(node)
		pred = sorted(pred, key=lambda k: network[k][node]['weight'], reverse=True)
		if len(pred) == 2 and (pred[1] in nx.ancestors(network, pred[0]) or pred[0] in nx.ancestors(network, pred[1])):
			print("CASE 1: X-Y->Z, X->Z")
			if pred[1] in nx.ancestors(network, pred[0]):
				network.remove_edge(pred[1], node)
			else:
				network.remove_edge(pred[0], node)
		else:	
			print("CASE 2: R->X->Z, R->Y->Z")
			for anc_node in pred[1:]:
				network.remove_edge(anc_node, node)
