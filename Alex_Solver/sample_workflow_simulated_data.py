import numpy as np
import random

from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
from simulation_tools.dataset_generation import generate_simulated_full_tree
from simulation_tools.simulation_utils import get_leaves_of_tree
from simulation_tools.validation import check_triplets_correct

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


# Generate dropout probabilities
data_dropout_rates = {24:0, 0: 0.14516129032258063, 1: 0.16129032258064513, 2: 0.12903225806451613, 3: 0.03225806451612903, 4: 0.03225806451612903, 5: 0.03225806451612903, 6: 0.08064516129032258, 9: 0.0967741935483871, 10: 0.11290322580645161, 11: 0.11290322580645161, 12: 0.11290322580645161, 13: 0.06451612903225806, 14: 0.06451612903225806, 15: 0.17741935483870963, 16: 0.17741935483870963, 17: 0.17741935483870963, 18: 0.22580645161290314, 19: 0.20967741935483863, 20: 0.19354838709677413, 21: 0.06451612903225806, 22: 0.06451612903225806, 23: 0.06451612903225806, 25: 0.12903225806451613, 26: 0.11290322580645161, 27: 0.11290322580645161, 28: 0.11290322580645161, 29: 0.11290322580645161, 30: 0.49999999999999967, 31: 0.35483870967741915, 32: 0.24193548387096764, 34: 0.0967741935483871, 35: 0.0967741935483871, 36: 0.11290322580645161, 37: 0.16129032258064513, 38: 0.06451612903225806, 39: 0.08064516129032258, 40: 0.06451612903225806, 41: 0.04838709677419355, 42: 0.46774193548387066, 43: 0.46774193548387066, 45: 0.8870967741935477, 46: 0.8870967741935477}
# New Dropouts data_dropout_rates = {0: 0.28489208633093527, 1: 0.28489208633093527, 2: 0.28489208633093527, 3: 0.10215827338129496, 4: 0.10215827338129496, 5: 0.10215827338129496, 6: 0.1712230215827338, 7: 0.1712230215827338, 8: 0.1712230215827338, 9: 0.29640287769784174, 10: 0.29640287769784174, 11: 0.29640287769784174, 12: 0.3597122302158273, 13: 0.3597122302158273, 14: 0.3597122302158273, 15: 0.3064748201438849, 16: 0.3064748201438849, 17: 0.3064748201438849, 18: 0.14244604316546763, 19: 0.14244604316546763, 20: 0.14244604316546763, 21: 0.6906474820143885, 22: 0.6906474820143885, 23: 0.6906474820143885, 24: 0.2935251798561151, 25: 0.2935251798561151, 26: 0.2935251798561151, 27: 0.7467625899280576, 28: 0.7467625899280576, 29: 0.7467625899280576, 30: 0.7697841726618705, 31: 0.7697841726618705, 32: 0.7697841726618705, 33: 0.11510791366906475, 34: 0.11510791366906475, 35: 0.11510791366906475, 36: 0.2589928057553957, 37: 0.2589928057553957, 38: 0.2589928057553957, 39: 0.03741007194244604, 40: 0.03741007194244604, 41: 0.03741007194244604}
dropout_prob_map = {i: random.choice(data_dropout_rates.values()) for i in range(0,40)}

# Generate simulated network
true_network = generate_simulated_full_tree(prior_probabilities, dropout_prob_map, characters=number_of_characters, subsample_percentage=.4, depth=9)

target_nodes = get_leaves_of_tree(true_network, clip_identifier=True)
target_nodes_original_network = get_leaves_of_tree(true_network, clip_identifier=False)

print "CCI complexity of reconstruction: ", cci_score(target_nodes)

# Hybrid solution
reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="greedy", prior_probabilities=prior_probabilities)

print "Number of triplets correct: ", check_triplets_correct(true_network, reconstructed_network_greedy)

#reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="ilp", prior_probabilities=prior_probabilities)

#rint "Number of triplets correct: ", check_triplets_correct(true_n/etwork, reconstructed_network_greedy)