import networkx as nx
import numpy as np

def node_parent(x, y):
	"""
	Given two nodes, finds the latest common ancestor

	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:return:
		Returns latest common ancestor of x and y
	"""

	parr = []
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

	return '|'.join(parr)

def get_edge_length(x,y,priors=None):
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
	count = 0
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
					count += - np.log(priors[i][str(y_list[i])])
			else:
				return -1
	return count

def mutations_from_parent_to_child(parent, child):
	"""
	Creates a string label describing the mutations taken from  a parent to a child
	:param parent: A node in the form 'Ch1|Ch2|....|Chn'
	:param child: A node in the form 'Ch1|Ch2|....|Chn'
	:return: A comma seperated string in the form Ch1: 0-> S1, Ch2: 0-> S2....
	where Ch1 is the character, and S1 is the state that Ch1 mutaated into
	"""
	parent_list = parent.split('|')
	child_list = child.split('|')
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
	np = target_nodes[0]
	for sample in target_nodes:
		np = node_parent(sample, np)

	return np

def build_potential_graph_from_base_graph_OLD(samples, priors=None):
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
	initial_network = nx.DiGraph()
	samples = set(samples)
	for sample in samples:
		initial_network.add_node(sample)

	samples = list(samples)

	source_nodes = samples

	print "Number of initial extrapolated pairs:", len(source_nodes)
	while len(source_nodes) != 1:
		temp_source_nodes = set()
		for i in range(0, len(source_nodes)-1):
			sample = source_nodes[i]
			top_parents = []
			for j in range(i + 1, len(source_nodes)):
				sample_2 = source_nodes[j]
				if sample != sample_2:
					parent = node_parent(sample, sample_2)
					top_parents.append((get_edge_length(parent, sample) + get_edge_length(parent, sample_2), parent, sample_2))

					# Check this cutoff
					if get_edge_length(parent, sample) + get_edge_length(parent, sample_2) < 3:
						initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors), label=mutations_from_parent_to_child(parent, sample_2))
						initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors), label=mutations_from_parent_to_child(parent, sample))
						temp_source_nodes.add(parent)

			min_distance = min(top_parents, key = lambda k: k[0])[0]
			lst = [(s[1], s[2]) for s in top_parents if s[0] <= min_distance]

			for parent, sample_2 in lst:
				initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors), label=mutations_from_parent_to_child(parent, sample_2))
				initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors), label=mutations_from_parent_to_child(parent, sample))
				temp_source_nodes.add(parent)

		source_nodes = list(temp_source_nodes)

		print "Next layer number of nodes:", len(source_nodes)

	return initial_network

def build_potential_graph_from_base_graph(samples, priors=None):
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

	prev_network = None
	flag = False
	for max_neighbor_dist in range(0, 15):
		initial_network = nx.DiGraph()
		samples = set(samples)
		for sample in samples:
			initial_network.add_node(sample)

		samples = list(samples)

		source_nodes = samples
		print "Num Neighbors considered: ", max_neighbor_dist
		print "Number of initial extrapolated pairs:", len(source_nodes)
		while len(source_nodes) != 1:
			if len(source_nodes) > 3000 and prev_network != None:
				return prev_network
			elif len(source_nodes) > 3000 and prev_network == None:
				flag = True
			elif len(source_nodes) > 2000:
				flag = True
			temp_source_nodes = set()
			for i in range(0, len(source_nodes)-1):
				sample = source_nodes[i]
				top_parents = []
				for j in range(i + 1, len(source_nodes)):
					sample_2 = source_nodes[j]
					if sample != sample_2:
						parent = node_parent(sample, sample_2)
						top_parents.append((get_edge_length(parent, sample) + get_edge_length(parent, sample_2), parent, sample_2))

						#Check this cutoff
						if get_edge_length(parent, sample) + get_edge_length(parent, sample_2) < max_neighbor_dist:
							initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors), label=mutations_from_parent_to_child(parent, sample_2))
							initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors), label=mutations_from_parent_to_child(parent, sample))
							temp_source_nodes.add(parent)

				min_distance = min(top_parents, key = lambda k: k[0])[0]
				lst = [(s[1], s[2]) for s in top_parents if s[0] <= min_distance]

				for parent, sample_2 in lst:
					initial_network.add_edge(parent, sample_2, weight=get_edge_length(parent, sample_2, priors), label=mutations_from_parent_to_child(parent, sample_2))
					initial_network.add_edge(parent, sample, weight=get_edge_length(parent, sample, priors), label=mutations_from_parent_to_child(parent, sample))
					temp_source_nodes.add(parent)
				if len(temp_source_nodes) > 3000 and prev_network != None:
					return prev_network
			source_nodes = list(temp_source_nodes)

			print "Next layer number of nodes:", len(source_nodes)
		prev_network = initial_network
		if flag:
			if not prev_network:
				return initial_network
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
	return [x for x in tree.nodes() if tree.in_degree(x)==0 ]
