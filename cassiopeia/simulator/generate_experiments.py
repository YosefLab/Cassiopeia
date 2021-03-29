import networkx as nx
import numpy as np
import pickle as pic
import random

import cassiopeia.critique.simulation_utils as sim_utils


def generate_tree_with_n_cells_exponential(
    output_path,
    num_cells,
    fitness_num_dist,
    fitness_strength_dist,
    sampling_proportion=0.2,
    relative_extinction_rate=0.15,
):
    birth_rate, death_rate = sim_utils.estimate_birth_death_rate(
        1, round(num_cells / sampling_proportion), relative_extinction_rate
    )

    tree, leaves = sim_utils.generate_birth_death(
        np.random.exponential(),
        np.random.exponential(),
        birth_rate,
        death_rate,
        fitness_num_dist,
        fitness_strength_dist,
        num_extant=round(num_cells / sampling_proportion),
    )
    sel_leaves = np.random.choice(leaves, num_cells, replace=False)

    to_remove = list(set(leaves) - set(sel_leaves))
    for node in to_remove:
        sim_utils.remove_and_prune(node, tree)
    sim_utils.collapse_unifurcations(tree)

    pic.dump(tree, open(output_path, "wb"))

    return tree


def overlay_mutations_exponential(
    topology_path,
    cm_path,
    network_path,
    ground_cm_path,
    state_distribution,
    num_cassettes=13,
    cassette_size=3,
    mutation_proportion=0.5,
    total_dropout_proportion=0.2,
    stochastic_dropout_proportion=0.1,
    double_resection_range=0.005,
):

    mut_rate = -np.log(1 - mutation_proportion)
    mut_cdf = lambda t: 1 - (np.exp(-t * mut_rate))
    silence_rate_scale = -np.log(
        (total_dropout_proportion - 1) / (stochastic_dropout_proportion - 1)
    )
    silence_cdf = lambda t: 1 - (np.exp(-t * silence_rate_scale))
    # silence_rate = [silence_rate_scale] * num_cassettes

    network = pic.load(open(topology_path, "rb"))

    leaves = sim_utils.overlay_mutation_continuous(
        network,
        num_cassettes * cassette_size,
        state_distribution,
        mut_cdf,
        cassette_size,
        double_resection_range,
        silence_cdf,
    )

    if ground_cm_path:
        ground_cm = sim_utils.get_character_matrix(network, leaves)
        ground_cm = ground_cm.astype(str)
        ground_cm.to_csv(ground_cm_path, sep="\t")

    sim_utils.add_heritable_dropout(network)
    sim_utils.add_stochastic_dropout(
        network, leaves, stochastic_dropout_proportion, cassette_size
    )
    dropout_cm = sim_utils.get_character_matrix(network, leaves)
    dropout_cm.to_csv(cm_path, sep="\t")

    pic.dump(network, open(network_path, "wb"))

    return network
