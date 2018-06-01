import networkx as nx
import random

def cci_score(nodes):
	"""
	Returns a score between 0.5 and 1, corresponding to the complexity of parsimony problem (Where 0.5 means most complex)
	:param nodes:
			A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		Score between 0.5 and 1, where the lower the score the more complex/dirty the parsimony problem is
	"""

	compatability_network = build_incompatability_graph(nodes)

	compatability_score = [
		nx.number_connected_components(compatability_network) / (1.0 * nx.number_of_nodes(compatability_network))]
	cci = [0]
	i = 1
	x = [x[0] for x in sorted(compatability_network.degree().items(), key=lambda x: x[1], reverse=True)][::-1]
	while nx.number_connected_components(compatability_network) != nx.number_of_nodes(compatability_network):
		cci.append(i)
		compatability_network.remove_edges_from(compatability_network.edges(x.pop()))
		compatability_score.append(
			nx.number_connected_components(compatability_network) / (1.0 * nx.number_of_nodes(compatability_network)))
		i += 1

	cci_score = 0
	if len(cci) == 1:
		return 1
	else:
		for i in range(0, len(compatability_score) - 1):
			cci_score += ((compatability_score[i] + compatability_score[i + 1]) / 2.0) / (1.0 * len(cci) - 1)

	return cci_score

def build_incompatability_graph(nodes):
	"""
	Generates a graph where nodes represent characters, and edges represent incompatibility between two characters

	:param nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		A compatibility graph where edges correspond to incompatible characters
	"""

	# Adding one node per character to networkx compatibility_network graph
	columns = map(''.join, zip(*nodes))
	compatability_network = nx.Graph()
	for i in range(0, len(columns)):
		compatability_network.add_node(i)

	# Generating positions for all possible characters and states
	sets = [set(column) for column in columns]
	dct = {i: {} for i in range(0, len(columns))}
	for i in range(0, len(columns)):
		for char in sets[i]:
			dct[i][char] = set([pos for pos, c in enumerate(columns[i]) if char == c])


	def look_for_edge(i, j):
		"""
		Sub-function which looks for incompatibility between two characters
		:param i:
			Character i
		:param j:
			Character j
		:return:
			True if there is incompatibility, else False
		"""
		if i != j:
			for item in sets[i]:
				for item2 in sets[j]:
					if item != '-' and item2 != '-' and item != '0' and item2 != '0':
						s1 = dct[i][item]
						s2 = dct[j][item2]
						s1_na = dct[i]['-'] if '-' in dct[i] else set()
						s2_na = dct[j]['-'] if '-' in dct[j] else set()
						if len(s1 & s2) == 0 or s1.issubset(s2.union(s2_na)) or s2.issubset(s1.union(s1_na)):
							pass
						else:
							return True

	# Generating compatibility graph edges
	for i in range(0, len(columns)):
		for j in range(0, len(columns)):
			if i != j:
				if look_for_edge(i, j):
					compatability_network.add_edge(i, j)

	return compatability_network




def random_walk(graph, node, steps=8):
	"""
	Given a graph and a specific node within the graph, returns a node after steps random edge jumps

	:param graph:
		A networkx graph
	:param node:
		Any node within the graph
	:param steps:
		The number of steps within the random walk
	:return:
		A node within the network
	"""
	if k == 0:
		return node
	else:
		return random_walk(graph, random.choice(G.out_edges(node))[1], k - 1)