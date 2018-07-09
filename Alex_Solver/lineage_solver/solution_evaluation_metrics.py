from collections import defaultdict

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

	compatability_network, cvi = build_incompatability_graph_and_violating_samples(nodes)

	compatability_score = [
		nx.number_connected_components(compatability_network) / (1.0 * nx.number_of_nodes(compatability_network))]

	cci = [0]
	iters = 1
	while nx.number_connected_components(compatability_network) != nx.number_of_nodes(compatability_network):
		x = [x[0] for x in sorted(compatability_network.degree().items(), key=lambda x: x[1], reverse=False)]
		cci.append(iters)
		compatability_network.remove_edges_from(compatability_network.edges(x.pop()))
		compatability_score.append(
			nx.number_connected_components(compatability_network) / (1.0 * nx.number_of_nodes(compatability_network)))
		iters += 1
	compatability_score += [1 for _ in range(0, nx.number_of_nodes(compatability_network) - len(compatability_score)+1)]

	cci_score = 0
	if len(cci) == 1:
		return 1
	else:
		for i in range(0, len(compatability_score) - 1):
			cci_score += ((compatability_score[i] + compatability_score[i + 1]) / 2.0) / (1.0 * len(cci) - 1)

	return cci_score

def get_cvi(nodes):
	"""
	Returns a dictionary whose keys correspond to samples, and whose values are sets of tuples, where the tuples represnts
		the two characters this sample caused incompatability

	:param nodes:
			A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		A dictionary whose keys correspond to samples, and whose values are sets of tuples, where the tuples represnts
		the two characters this sample caused incompatability for
	"""

	compatability_network, cvi = build_incompatability_graph_and_violating_samples(nodes)

	return cvi

def build_incompatability_graph_and_violating_samples(nodes):
	"""
	Generates a graph where nodes represent characters, and edges represent incompatibility between two characters
	Also returns a dictionary containing for each sample, all characters which were incompatable as a result of the
	presence of such sample

	:param nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		A compatibility graph where edges correspond to incompatible characters
		A dictionary whose keys correspond to samples, and whose values are sets of tuples, where the tuples represnts
		the two characters this sample caused incompatability for
	"""

	# Adding one node per character to networkx compatibility_network graph
	new_nodes = []
	for node in nodes:
		new_nodes.append(node.split('|'))
	zipped_columns = zip(*new_nodes)

	columns = map('|'.join, zip(*new_nodes))

	compatability_network = nx.Graph()

	for i in range(0, len(columns)):
		compatability_network.add_node(i)

	# Generating positions for all possible characters and states
	sets = [set(column.split('|')) for column in columns]
	dct = {i: {} for i in range(0, len(columns))}

	for i in range(0, len(columns)):
		for char in sets[i]:
			dct[i][char] = set([pos for pos, c in enumerate(zipped_columns[i]) if char == c])

	samples_dict = defaultdict(set)
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
		flag = False
		if i != j:
			for item in sets[i]:
				for item2 in sets[j]:
					if item != '-' and item2 != '-' and item != '0' and item2 != '0':
						s1 = dct[i][item]
						s2 = dct[j][item2]

						s1_na = dct[i]['-'] if '-' in dct[i] else set()
						s2_na = dct[j]['-'] if '-' in dct[j] else set()
						if (len(s1 & s2) == 0 or s1.issubset(s2.union(s2_na)) or s2.issubset(s1.union(s1_na))):
							pass
						else:
							#print i, j, item, item2, s1, s2, len(s1 & s2)
							# Revisit this section
							set1 = s1 & s2
							set2 = s1 - s2.union(s2_na)
							set3 = s2 - s1.union(s1_na)

							if len(set1) == min(len(set1), len(set2), len(set3)):
								ultimate_set = set1
							elif len(set2) == min(len(set1), len(set2), len(set3)):
								ultimate_set = set2
							else:
								ultimate_set = set3

							for node in ultimate_set:
								samples_dict[node].add((i,j))

							flag = True
			if flag:
				return True

	# Generating compatibility graph edges

	for i in range(0, len(columns)):
		for j in range(0, len(columns)):
			if i != j:
				if look_for_edge(i, j):
					compatability_network.add_edge(i, j)
		

	return compatability_network, samples_dict

def flag_double_mutated_samples(nodes, character, state):
	"""
	Given a set of nodes, as well as a character and state of interest, find samples which did not mutate off of the main mutation branch
	(i.e. the mutation occured independently from the main mutation branch)
	:param nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param character:
		Character of interest as an integer
	:param state:
		The state of interest as a string
	:return: A set of samples believed to be either with error, or having mutated off an off-branch
	"""


	# Converting nodes to column vectors
	new_nodes = []
	for node in nodes:
		new_nodes.append(node.split('|'))
	zipped_columns = zip(*new_nodes)

	columns = map('|'.join, zip(*new_nodes))


	# Generating positions for all possible characters and states
	sets = [set(column.split('|')) for column in columns]
	dct = {i: {} for i in range(0, len(columns))}

	for i in range(0, len(columns)):
		for char in sets[i]:
			dct[i][char] = set([pos for pos, c in enumerate(zipped_columns[i]) if char == c])


	pre_filtered_samples = dct[character][str(state)]

	# Loops through all other characters, and looks at the minimal compatability set
	# Based on this set, it flags samples if they deviate too much from the primary set of samples
	for i in range(0, len(columns)):
		for item in sets[i]:
			seen = set()
			if item != '-' and item != '0':
				s1 = dct[i][item]
				s2 = dct[character][state]
				s1_na = dct[i]['-'] if '-' in dct[i] else set()
				s2_na = dct[character]['-'] if '-' in dct[character] else set()

				if (len(s1 & s2) == 0 or s1.issubset(s2.union(s2_na)) or s2.issubset(s1.union(s1_na))):
					pass
				else:
					print i, item
					# print i, j, item, item2, s1, s2, len(s1 & s2)
					set1 = s1.union(s1_na) & s2.union(s2_na)
					set2 = s1 - s2.union(s2_na)
					set3 = s2 - s1.union(s1_na)

					if len(set2) > 10:

						character_set_similiarity = 0
						second_character_similarity = 0
						if len(set1) > len(set3) and len(set2) > len(set3):
							for element in set3:
								sample = nodes[element].split('|')
								for j in range(0, len(sample)):
									if not (sample[j], j) in seen:
										if sample[j] != '0' and sample[j] != '-' and j != character and j != i:

											character_set_similiarity += len(dct[j][sample[j]] & set1 )/(1.0 * len(set1) *len(sample))
											second_character_similarity += len(dct[j][sample[j]] & set2 )/ (1.0 * len(set2)*len(sample))
											seen.add((sample[j], j))

							if character_set_similiarity < second_character_similarity - .075:
								pre_filtered_samples = pre_filtered_samples - set3

						elif len(set3) > len(set1) and len(set2) > len(set1):
							for element in set1:
								sample = nodes[element].split('|')
								for j in range(0, len(sample)):
									if not (sample[j], j) in seen:
										if sample[j] != '0' and sample[j] != '-' and j != character and j != i:
											character_set_similiarity += len(dct[j][sample[j]] & set3) / (1.0 * len(set3)*len(sample))
											second_character_similarity += len(dct[j][sample[j]] & set2) / (1.0 * len(set2)*len(sample))
											seen.add((sample[j], j))
							if character_set_similiarity < second_character_similarity-.075:
								pre_filtered_samples = pre_filtered_samples & (s2 - s1)


	return  dct[character][str(state)] - pre_filtered_samples

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
	if steps == 0:
		return node
	else:
		return random_walk(graph, random.choice(graph.out_edges(node))[1], steps - 1)