import networkx as nx
import numpy as np

from Cassiopeia.TreeSolver import Node

def node_parent(n1, n2):
	"""
	Given two nodes, finds the latest common ancestor

	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:return:
		Returns latest common ancestor of x and y
	"""
	x, y = n1.get_character_string(), n2.get_character_string()

	parr = []
	if '_' in x:
		x = ''.join(x.split("_")[:-1])
	if '_' in y:
		y = ''.join(y.split("_")[:-1])

	x_list = x.split('|')
	y_list = y.split('|')
	for i in range(0,len(x_list)):
		if x_list[i] == y_list[i]:
			parr.append(x_list[i])
		elif x_list[i] == '-':
			parr.append(y_list[i])
		elif y_list[i] == '-':
			parr.append(x_list[i])
		else:
			parr.append('0')

	parent_node = Node("internal", parr, is_target=False)

	return parent_node

def get_edge_length(n1,n2,priors=None):
	"""
	Given two nodes, if x is a parent of y, returns the edge length between x and y, else -1

	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:param priors:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		Length of edge if valid transition, else -1
	"""

	x, y = n1.get_character_string(), n2.get_character_string()

	count = 0
	if '_' in x:
		x = ''.join(x.split("_")[:-1])
	if '_' in y:
		y = ''.join(y.split("_")[:-1])
	x_list = x.split('|')
	y_list = y.split('|')

	for i in range(0, len(x_list)):
		if x_list[i] == y_list[i]:
			pass
		elif y_list[i] == "-":
			count += 0

		elif x_list[i] == '0':
			if not priors:
				count += 1
			else:
				count += -np.log(priors[i][str(y_list[i])])
		else:
			return -1
	return count

def mutations_from_parent_to_child(p_node, c_node):
	"""
	Creates a string label describing the mutations taken from  a parent to a child
	:param parent: A node in the form 'Ch1|Ch2|....|Chn'
	:param child: A node in the form 'Ch1|Ch2|....|Chn'
	:return: A comma seperated string in the form Ch1: 0-> S1, Ch2: 0-> S2....
	where Ch1 is the character, and S1 is the state that Ch1 mutaated into
	"""

	parent, child = p_node.get_character_string(), c_node.get_character_string()

	if '_' in parent:
		parent = ''.join(parent.split("_")[:-1])
	if '_' in child:
		child = ''.join(child.split("_")[:-1])

	parent_list = parent.split("_")[0].split('|')
	child_list = child.split("_")[0].split('|')
	mutations = []
	for i in range(0, len(parent_list)):
		if parent_list[i] != child_list[i] and child_list[i] != '-':
			mutations.append(str(i) + ": " + str(parent_list[i]) + "->" + str(child_list[i]))

	return " , ".join(mutations)

def root_finder(target_nodes):
	"""
	Given a list of targets_nodes, return the least common ancestor of all nodes

	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		The least common ancestor of all target nodes, in the form 'Ch1|Ch2|....|Chn'
	"""

	n = target_nodes[0]
	for sample in target_nodes:
		n = node_parent(sample, n)

	return n

def build_potential_graph_from_base_graph(samples, root, max_neighborhood_size = 10000, priors=None, pid=-1):
	"""
	Given a series of samples, or target nodes, creates a tree which contains potential
	ancestors for the given samples.

	First, a directed graph is constructed, by considering all pairs of samples, and checking
	if a sample can be a possible parent of another sample
	Then we all pairs of nodes with in-degree 0 and < a certain edit distance away
	from one another, and add their least common ancestor as a parent to these two nodes. This is done
	until only one possible ancestor remains

	:param samples:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param priors
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		A graph, which contains a tree which explains the data with minimal parsimony
	"""
		#print "Initial Sample Size:", len(set(samples))

	cdef int neighbor_mod
	cdef int max_dist

	neighbor_mod = 0
	prev_network = None
	flag = False
	max_dist = len(samples[0].get_character_vec())

	for max_neighbor_dist in range(0, min(max_dist, 14)):
		initial_network = nx.DiGraph()
		samples = set(samples)
		for sample in samples:
			initial_network.add_node(sample)

		samples = list(samples)

		source_nodes = samples
		neighbor_mod = max_neighbor_dist
		char_string_edges = []
		char_strings_to_node = dict(zip([s.get_character_string() for s in samples], samples))


		print("\nNum Neighbors considered: " + str(max_neighbor_dist) + ", pid = " + str(pid))
		print("Number of initial extrapolated pairs:" + str(len(source_nodes)) + ", pid = " + str(pid))

		while len(source_nodes) != 1:

			if len(source_nodes) > int(max_neighborhood_size):
				print("Max Neighborhood Exceeded, Returning Network")
				return prev_network

			temp_source_nodes = set()
			for i in range(0, len(source_nodes)-1):

				sample = source_nodes[i]
				top_parents = []
				p_to_s1_lengths, p_to_s2_lengths = {}, {}
				muts_to_s1, muts_to_s2 = {}, {}

				for j in range(i + 1, len(source_nodes)):
					sample_2 = source_nodes[j]
					
					if sample.get_character_string() != sample_2.get_character_string():

						parent = node_parent(sample, sample_2)

						# if parent already exists, we need to find it
						if parent.get_character_string() in list(char_strings_to_node.keys()):
							parent = char_strings_to_node[parent.get_character_string()]

						if parent != root:
							parent.pid = pid

						edge_length_p_s1 = get_edge_length(parent, sample)
						edge_length_p_s2 = get_edge_length(parent, sample_2)
						top_parents.append((edge_length_p_s1 + edge_length_p_s2, parent, sample_2))

						muts_to_s1[(parent, sample)] = mutations_from_parent_to_child(parent, sample)
						muts_to_s2[(parent, sample_2)] = mutations_from_parent_to_child(parent, sample_2)

						p_to_s1_lengths[(parent, sample)] = edge_length_p_s1
						p_to_s2_lengths[(parent, sample_2)] = edge_length_p_s2

						#Check this cutoff
						if edge_length_p_s1 + edge_length_p_s2 < neighbor_mod:

							edge_length_p_s1_priors, edge_length_p_s2_priors = get_edge_length(parent, sample, priors), get_edge_length(parent, sample_2, priors)

							if parent.get_character_string() != sample_2.get_character_string() and (parent.get_character_string(), sample_2.get_character_string()) not in char_string_edges:
								initial_network.add_edge(parent, sample_2, weight=edge_length_p_s2_priors, label=muts_to_s2[(parent, sample_2)])
								char_string_edges.append((parent.get_character_string(), sample_2.get_character_string()))

							if parent.get_character_string() != sample.get_character_string() and (parent.get_character_string(), sample.get_character_string()) not in char_string_edges:
								initial_network.add_edge(parent, sample, weight=edge_length_p_s1_priors, label=muts_to_s1[(parent, sample)])
								char_string_edges.append((parent.get_character_string(), sample.get_character_string()))


							temp_source_nodes.add(parent)
							char_strings_to_node[parent.get_character_string()] = parent


							p_to_s1_lengths[(parent, sample)] = edge_length_p_s1_priors
							p_to_s2_lengths[(parent, sample_2)] = edge_length_p_s2_priors

				min_distance = min(top_parents, key = lambda k: k[0])[0]
				lst = [(s[1], s[2]) for s in top_parents if s[0] <= min_distance]

				for parent, sample_2 in lst:

					if parent.get_character_string() != sample_2.get_character_string() and (parent.get_character_string(), sample_2.get_character_string()) not in char_string_edges:
						initial_network.add_edge(parent, sample_2, weight=p_to_s2_lengths[(parent, sample_2)], label=muts_to_s2[(parent, sample_2)])
						char_string_edges.append((parent.get_character_string(), sample_2.get_character_string()))

					if parent.get_character_string() != sample.get_character_string() and (parent.get_character_string(), sample.get_character_string()) not in char_string_edges:
						initial_network.add_edge(parent, sample, weight=p_to_s1_lengths[(parent, sample)], label=muts_to_s1[(parent, sample)])
						char_string_edges.append((parent.get_character_string(), sample.get_character_string()))

					if parent.get_character_string() not in [n.get_character_string() for n in temp_source_nodes]:
						temp_source_nodes.add(parent)

				if len(temp_source_nodes) > int(max_neighborhood_size) and prev_network != None:
					return prev_network
			if len(source_nodes) > len(temp_source_nodes):
				if neighbor_mod == max_neighbor_dist:
					neighbor_mod *= 3
			source_nodes = list(temp_source_nodes)
			print("Next layer number of nodes: " + str(len(source_nodes)) + " - pid = " + str(pid))

		#print('testing isomporpic!')
		#if prev_network is not None and nx.graph_edit_distance(prev_network, initial_network) == 0:
		#if prev_network is not None and nx.is_isomorphic(prev_network, initial_network):
		#	return prev_network

		prev_network = initial_network
		if flag:
			return prev_network

	return initial_network


def get_sources_of_graph(tree):
	"""
	Returns all nodes with in-degree zero

	:param tree:
		networkx tree
	:return:
		Leaves of the corresponding Tree
	"""
	return [x for x in tree.nodes() if tree.in_degree(x)==0]
