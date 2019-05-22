import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from simulation_utils import node_to_string
from Cassiopeia.TreeSolver.Node import Node
from Cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

def generate_simulated_full_tree(mutation_prob_map, variable_dropout_prob_map, characters=10, depth=12, subsample_percentage = 0.1, dropout=True):
	"""
	Given the following parameters, this method simulates the cell division and mutations over multiple lineages
		- Cells/Samples are treated as a string, with a unique identifier appended to the end of the string,
		  in the form sample = 0|3|0|12, where len(sample.split('|')) = characters
		- Each generation, all cells are duplicated, and each character is independently transformed
      	  with the probabilities of transformation defined in mutation_prob_map
		- At the end of this process of duplication, there will be 2 ^ depth samples.
		- We sample subsample_percentage of the 2 ^ depth samples
		- On the subsampled population, we simulate dropout on each individual character in each sample
		  with probability variable_dropout_prob_map


	:param mutation_prob_map:
		A nested dictionary containing mutation probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
		I.e {0:{"0":0.975, "1":0.25},....}
	:param variable_dropout_prob_map:
		A dictionary containing dropout probabilities for each individual character
		I.e {0: 0.05, 1: 0.01, 2: 0.2,...}
	:param characters:
		The number of characters to simulate
	:param depth:
		Number of generations to apply the above process
	:param subsample_percentage:
		Percentage of population to subsample after the final generation
	:return:
		A networkx tree of samples
	"""

	network = nx.DiGraph()
	current_depth = [[['0' for _ in range(0, characters)], '0']]
	network.add_node(node_to_string(current_depth[0]))
	uniq = 1
	for i in range(0, depth):
		temp_current_depth = []
		for node in current_depth:
			for _ in range(0,2):
				child_node = simulate_mutation(node[0], mutation_prob_map)
				if i == depth - 1 and dropout:
					child_node = simulate_dropout(child_node, variable_dropout_prob_map)
				temp_current_depth.append([child_node, uniq])
				network.add_edge(node_to_string(node), node_to_string([child_node, str(uniq)]))
				uniq +=1

		current_depth = temp_current_depth

	subsampled_population_for_removal = random.sample(current_depth, int((1-subsample_percentage) * len(current_depth)))

	for node in subsampled_population_for_removal:
		network.remove_node(node_to_string(node))

	rdict = {}
	i = 0
	for n in network.nodes:
		nn = Node("StateNode" + str(i), n.split("_")[0].split("|"), pid = n.split("_")[1], is_target=False)
		i += 1
		rdict[n] = nn

	state_tree = nx.relabel_nodes(network, rdict)


	return state_tree

def generate_simulated_ivlt_experiment(mutation_prob_map, variable_dropout_prob_map, characters=10, gen_per_dish=7, num_splits = 2, subsample_percentage = 0.1):

	network = nx.DiGraph()
	current_depth = [[['0' for _ in range(0, characters)], "0"]]
	network.add_node(node_to_string(current_depth[0]))
	network.node[node_to_string(current_depth[0])]["plate"] = ""
	uniq = 1


    #simulate two splits total
	total_depth = (num_splits + 1) * gen_per_dish
	for i in tqdm(range(0, total_depth), desc="Generating cells at each level in tree"):
		temp_current_depth = []
		for node in current_depth:
			for _ in range(0, 2):
				child_node = simulate_mutation(node[0], mutation_prob_map)
				if i == total_depth - 1:
					child_node = simulate_dropout(child_node, variable_dropout_prob_map)

				temp_current_depth.append([child_node, uniq])
				network.add_edge(node_to_string(node), node_to_string([child_node, str(uniq)]))
				if i != 0 and i % gen_per_dish == 0:
					split_right = (np.random.random() > 0.5) # split in half
					if split_right:
						network.node[node_to_string([child_node, str(uniq)])]["plate"] = network.node[node_to_string(node)]["plate"] + "0"
					else:
						network.node[node_to_string([child_node, str(uniq)])]["plate"] = network.node[node_to_string(node)]["plate"] + "1"

				else:
					network.node[node_to_string([child_node, str(uniq)])]["plate"] = network.node[node_to_string(node)]["plate"]

				uniq += 1


		current_depth = temp_current_depth

	subsampled_population_for_removal = random.sample(current_depth, int((1-subsample_percentage) * len(current_depth)))

	for node in subsampled_population_for_removal:
		network.remove_node(node_to_string(node))

	return network



def simulate_dropout(sample, variable_dropout_probability_map):
	"""
	Applies dropout to a given sample

	:param sample:
		Samples in list form: I.e. ['0','1','0','1']
	:param variable_dropout_prob_map:
		A dictionary containing dropout probabilities for each individual character
		I.e {0: 0.05, 1: 0.01, 2: 0.2,...}
	:return:
		A sample with characters potential dropped out (Dropped out characters in the form '-')
	"""
	new_sample = []
	for i in range(0, len(sample)):
		if random.uniform(0, 1) <= variable_dropout_probability_map[i]:
			new_sample.append('-')
		else:
			new_sample.append(sample[i])
	return new_sample

def simulate_mutation(sample, mutation_prob_map):
	"""
	Transforms a newly generated sample into a potentially mutated one

	:param sample:
		Samples in list form: I.e. ['0','1','0','1']
	:param mutation_prob_map:
		A nested dictionary containing mutation probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
		I.e {0:{"0":0.975, "1":0.25},....}

	:return:
		A sample with characters potential mutated
	"""
	new_sample = []
	for i in range(0, len(sample)):
		character = sample[i]
		if character == '0':
			values, probabilities = zip(*mutation_prob_map[i].items())
			new_character = np.random.choice(values, p=probabilities)
			new_sample.append(new_character)
		else:
			new_sample.append(character)
	return new_sample
