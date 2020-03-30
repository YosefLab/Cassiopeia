import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from simulation_utils import node_to_string, get_leaves_of_tree
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree


def generate_simulated_full_tree(
    mutation_prob_map,
    variable_dropout_prob_map,
    characters=10,
    depth=12,
    subsample_percentage=0.1,
    dropout=True,
):
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
    current_depth = [[["0" for _ in range(0, characters)], "0"]]
    network.add_node(node_to_string(current_depth[0]))
    uniq = 1
    for i in range(0, depth):
        temp_current_depth = []
        for node in current_depth:
            for _ in range(0, 2):
                child_node = simulate_mutation(node[0], mutation_prob_map)
                if i == depth - 1 and dropout:
                    child_node = simulate_dropout(child_node, variable_dropout_prob_map)
                temp_current_depth.append([child_node, uniq])
                network.add_edge(
                    node_to_string(node), node_to_string([child_node, str(uniq)])
                )
                uniq += 1

        current_depth = temp_current_depth

    subsampled_population_for_removal = random.sample(
        current_depth, int((1 - subsample_percentage) * len(current_depth))
    )

    for node in subsampled_population_for_removal:
        network.remove_node(node_to_string(node))

    rdict = {}
    i = 0
    for n in network.nodes:
        nn = Node(
            "StateNode" + str(i),
            n.split("_")[0].split("|"),
            pid=n.split("_")[1],
            is_target=False,
        )
        i += 1
        rdict[n] = nn

    state_tree = nx.relabel_nodes(network, rdict)

    state_tree = Cassiopeia_Tree("simulated", network=state_tree)

    return state_tree


def generate_simulated_experiment_plasticisity(
    mutation_prob_map,
    variable_dropout_prob_map,
    characters=10,
    subsample_percentage=0.1, # rate at which to subsample leaves 
    dropout=True, # perform dropout on cells 
    sample_list = ['A'], # set of possible labels for each cell
    max_mu = 0.3, # maximum rate of change
    min_alpha = 0.0, # minimum rate of change
    depth = 11, # depth of tree, or number of time steps
    beta = 0, # extinction rate 
):

    mu = max_mu*np.random.random() # probability of change per time-step (between 0 and 0.3)
    alpha = min_alpha+((1-min_alpha)*np.random.random()) # probability that cell will double per time-step (between 0.75 and 1.0)

    n_transitions = 0

    network = nx.DiGraph()
    current_depth = [Node('state-node', character_vec=["0" for _ in range(0, characters)])]
    # current_depth = [[["0" for _ in range(0, characters)], "0"]]
    network.add_node(current_depth[0])
    network.node[current_depth[0]]["meta"] = sample_list[0]
    uniq = 1

    for i in tqdm(range(0, depth), desc="Generating cells at each level in tree"):
        temp_current_depth = []
        for node in current_depth:

            if np.random.random() < alpha:
                for _ in range(0, 2):
                    child_node = simulate_mutation(node.char_vec, mutation_prob_map)
                    if i == depth - 1:
                        child_node = simulate_dropout(child_node, variable_dropout_prob_map)

                    child_node = Node('state-node', character_vec = child_node)

                    temp_current_depth.append(child_node)
                    network.add_edge(
                        node, child_node
                    )

                    if np.random.random() < mu:
                        temp_sample_list = sample_list.copy()
                        temp_sample_list.remove(network.node[node]['meta'])
                        trans = np.random.choice(temp_sample_list)
                        network.node[child_node]['meta'] = trans
                        n_transitions += 1

                    else: 
                        network.node[child_node]['meta'] = network.node[node]['meta']
                
                    uniq += 1
            else:
                temp_current_depth.append(node)
                
        current_depth = temp_current_depth

    # # subsample nodes
    # subsampled_population_for_removal = random.sample(
    #     current_depth, int((1 - subsample_percentage) * len(current_depth))
    # )

    # print('removing ' + str(len(subsampled_population_for_removal)) + ' during subsampling of ' + str(len(current_depth)) + ' cells')

    # # subsample leaves
    # for node in subsampled_population_for_removal:
    #     network.remove_node(node)

    # # check back edges that don't lead to a leaf
    # targets = get_leaves_of_tree(network)
    # all_leaves = [n for n in network if network.out_degree(n) == 0]
    # for l in all_leaves:
    #     if l not in targets:
    #         node = l
    #         while network.out_degree(node) == 0:
    #             parent = list(network.predecessors(node))[0]
    #             network.remove_node(node)
    #             node = parent

    # rename nodes for easy lookup later 
    i = 0
    for n in network.nodes:
        n.name = 'StateNode' + str(i)
        i += 1

    params = {'mu': mu, 
            'alpha': alpha, 
            'N': len([n for n in network if network.out_degree(n) == 0])
    }

    tree = Cassiopeia_Tree("simulated", network=network)
    return tree, params


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
            new_sample.append("-")
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
        if character == "0":
            values, probabilities = zip(*mutation_prob_map[i].items())
            new_character = np.random.choice(values, p=probabilities)
            new_sample.append(new_character)
        else:
            new_sample.append(character)
    return new_sample
