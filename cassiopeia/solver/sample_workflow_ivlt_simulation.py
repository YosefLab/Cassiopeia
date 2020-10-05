import numpy as np
import networkx as nx
import random

import pickle as pic

from simulation_tools.dataset_generation import generate_simulated_ivlt_experiment
from simulation_tools.simulation_utils import get_leaves_of_tree

from data_pipeline import convert_network_to_newick_format

# Generate mutation probabilities
number_of_characters = 40
number_of_states = 10
prior_probabilities = {}
for i in range(0, number_of_characters):
	sampled_probabilities = sorted([np.random.negative_binomial(5,.5) for _ in range(1,number_of_states)])
	prior_probabilities[i] = {'0':.975}
	total = np.sum(sampled_probabilities)
	for j in range(1, number_of_states):
		prior_probabilities[i][str(j)] = (1-0.975)*sampled_probabilities[j-1]/(1.0 * total)

dropout_prob_map = {i: 0.0 for i in range(0, 40)}

# Generate simulated network
true_network = generate_simulated_ivlt_experiment(prior_probabilities, dropout_prob_map, characters=number_of_characters, subsample_percentage=1, gen_per_dish=5)

target_nodes = get_leaves_of_tree(true_network, clip_identifier=True)
target_nodes_original_network = get_leaves_of_tree(true_network, clip_identifier=False)

pic.dump([true_network.nodes(), true_network.edges()], open("simulated_ivlt.pkl", "wb"))

# write out meta
with open("simulated_meta.txt", "w") as f:
    f.write("Name\tPlateID\tFirstSplit\n")
    for node in target_nodes_original_network:
        f.write(str(node) + "\t" + str(true_network.node[node]["plate"]) + "\t")
        pid = true_network.node[node]["plate"]
        fs = '0' if (pid == '00' or pid == '01' or pid == '0') else '1'
        f.write(str(fs) + "\n")

# write out character matrix
with open("simulated_X.txt", "w") as f:
    # write header
    f.write("cellBC")
    num_char = len(target_nodes[0].split("|"))
    for n in range(num_char):
        f.write("\tr" + str(n + 1))
    f.write("\n")
    
    for sample in target_nodes_original_network:
        
        f.write(sample)
        chars = sample.split("_")[0].split("|")
        for c in chars:
            f.write("\t" + str(c))

        f.write("\n")

newick = convert_network_to_newick_format(true_network)
with open("simulated_ivlt.txt", "w") as f:
    f.write(newick)
