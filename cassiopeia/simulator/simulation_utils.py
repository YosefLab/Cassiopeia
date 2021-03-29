from Bio import Phylo as Phylo
from io import StringIO
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pic
import random
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from typing import List

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)

# utils.install_packages('mnormt')
# utils.install_packages('TreeSimGM')
# utils.install_packages('ape')


def get_character_matrix(
    network: nx.DiGraph, leaves: List[int]
) -> pd.DataFrame:

    nodes_to_characters = {}
    for node in leaves:
        character_vector = network.nodes[node]["characters"]
        character_vector[node] = character_vector

    return pd.DataFrame.from_dict(nodes_to_characters)


def prune_directed_graph(network):
    root = [a for a in network.nodes()][0]
    if network.out_degree(root) > 0:
        for node in network.successors(root):
            directed_graph_helper(network, node, root)


def directed_graph_helper(network, node, parent):
    network.remove_edge(node, parent)
    for i in network.successors(node):
        directed_graph_helper(network, i, node)


def estimate_birth_death_rate(time, n, epsilon):
    ret = []
    test = robjects.r(
        """
    epsilon = """
        + str(epsilon)
        + """
    r = bd.ms(time = """
        + str(time)
        + """, n = """
        + str(n)
        + """, epsilon = epsilon, crown = TRUE)
    lambda = r/(1-epsilon)
    lambda
    """
    )
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)

    test = robjects.r(
        """
    epsilon = """
        + str(epsilon)
        + """
    r = bd.ms(time = """
        + str(time)
        + """, n = """
        + str(n)
        + """, epsilon = epsilon, crown = TRUE)
    lambda = r/(1-epsilon)
    mu = lambda * epsilon
    mu
    """
    )
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)
    return ret


def remove_and_prune(node, network):
    curr_parent = list(network.predecessors(node))[0]
    network.remove_node(node)
    while (
        network.out_degree(curr_parent) < 1
        and network.in_degree(curr_parent) > 0
    ):
        next_parent = list(network.predecessors(curr_parent))[0]
        network.remove_node(curr_parent)
        curr_parent = next_parent


def construct_topology(
    div_rate,
    death_rate,
    fitness=None,
    fit_base=None,
    num_leaves=None,
    time=None,
):

    if fitness == 0 or fit_base == 1:
        fitness = None
        fit_base = None

    if time == None and num_leaves:
        if fitness == None:
            test = robjects.r(
                """
                test <- function ()
                {
                    yule <- sim.taxa(1, n="""
                + str(num_leaves)
                + """,
                    waitsp="rexp("""
                + str(div_rate)
                + """)", 
                    waitext="rexp("""
                + str(death_rate)
                + """)", complete = FALSE)
                    write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                }
                test()
                """
            )
        else:
            test = robjects.r(
                """
                test <- function ()
                {
                    yule <- sim.taxa(1, n="""
                + str(num_leaves)
                + """,
                    waitsp="rexp("""
                + str(div_rate)
                + """)", 
                    waitext="rexp("""
                + str(death_rate)
                + """)",
                    shiftsp=list(prob="""
                + str(fitness)
                + ''', strength="'''
                + str(fit_base)
                + """**runif(-1,1)"),
                    complete = FALSE)
                    write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                }
                test()
                """
            )

    elif num_leaves == None and time:
        try:
            if fitness == None:
                test = robjects.r(
                    """
                    test <- function ()
                    {
                        yule <- sim.age(1, age="""
                    + str(time)
                    + """, 
                        waitsp="rexp("""
                    + str(div_rate)
                    + """)", 
                        waitext="rexp("""
                    + str(death_rate)
                    + """)", complete = FALSE)
                        write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                    }
                    test()
                    """
                )
            else:
                test = robjects.r(
                    """
                    test <- function ()
                    {
                        yule <- sim.age(1, age="""
                    + str(time)
                    + """, 
                        waitsp="rexp("""
                    + str(div_rate)
                    + """)", 
                        waitext="rexp("""
                    + str(death_rate)
                    + """)",
                        shiftsp=list(prob="""
                    + str(fitness)
                    + ''', strength="'''
                    + str(fit_base)
                    + """**runif(-1,1)"),
                        complete = FALSE)
                        write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                    }
                    test()
                    """
                )
        except Exception as e:
            if e.__class__.__name__ == "RRuntimeError":
                print(
                    "R Error: likely that lineage fully deceased by the end of experiment, no samples generated"
                )
                return None, []

    else:
        print(
            "Please specify either a time length of experiment or a number of cells desired"
        )
        return

    for string in test.items():
        string

    tree = Phylo.read(StringIO(string[1]), "newick")
    network = Phylo.to_networkx(tree)
    network = network.to_directed()
    prune_directed_graph(network)

    leaves = [n for n in network if network.out_degree(n) == 0]

    rdict = {}
    i = 0
    for node in leaves:
        i += 1
        rdict[node] = i

    network = nx.relabel_nodes(network, rdict)

    leaves = [n for n in network if network.out_degree(n) == 0]
    return network, leaves


def overlay_mutation_continuous(
    network,
    num_chars,
    state_distribution,
    basal_rate,
    cassette_size,
    epsilon,
    silence_rates,
):
    for node in network.nodes:
        network.nodes[node]["characters"] = [0] * num_chars

    root = [n for n in network if network.in_degree(n) == 0][0]
    network.nodes[root]["parent_lifespan"] = 0

    mutation_helper_continuous(
        network,
        root,
        basal_rate,
        state_distribution,
        root.char_vec,
        set(),
        cassette_size,
        epsilon,
        silence_rates,
    )

    leaves = [n for n in network if network.out_degree(n) == 0]

    return leaves


def mutation_helper_continuous(
    network,
    node,
    mutation_cdf,
    mutation_prob_map,
    curr_mutations,
    dropout_indices,
    cassette_size,
    epsilon,
    silence_cdf,
):
    new_sample = curr_mutations.copy()
    new_dropout_indices = dropout_indices.copy()
    t = network.nodes[node]["parent_lifespan"]

    # p = 1 - (np.exp(-t * basal_rate))
    p = mutation_cdf(t)

    base_chars = []
    for i in range(0, len(new_sample)):
        if new_sample[i] == "0" and i not in new_dropout_indices:
            base_chars.append(i)

    draws = np.random.binomial(len(base_chars), p)
    chosen_ind = np.random.choice(base_chars, draws, replace=False)
    cassettes = {}

    for i in chosen_ind:
        values, probabilities = zip(*mutation_prob_map[i].items())
        new_character = np.random.choice(values, p=probabilities)
        new_sample[i] = new_character
        time = np.random.uniform(0.0, t)
        left = max(0, time - epsilon)
        right = min(t, time + epsilon)
        cass_num = i // cassette_size
        if cass_num in cassettes:
            cassettes[cass_num].append((left, right, i))
        else:
            cassettes[cass_num] = [(left, right, i)]

    for cass_num in cassettes.keys():
        if len(cassettes[cass_num]) > 1:
            time_ranges = []
            for cut in cassettes[cass_num]:
                time_ranges.append(cut)
            time_ranges.sort(key=lambda x: x[0])

            seen = set()
            for cut in time_ranges:
                if cut[2] in seen:
                    continue
                for cut2 in time_ranges:
                    if cut2[2] in seen:
                        continue
                    if cut[1] >= cut2[0]:
                        if cut[2] != cut2[2]:
                            for e in range(
                                min(cut[2], cut2[2]), max(cut[2], cut2[2]) + 1
                            ):
                                if e not in new_dropout_indices:
                                    new_dropout_indices.add(e)
                                    seen.add(e)
                                    seen.add(e)
                            break
                seen.add(cut[2])

    for i in range(len(silence_rates)):
        # silence_prob = 1 - (np.exp(-t * silence_rates[i]))
        silence_prob = silence_cdf(t)
        if random.uniform(0, 1) < silence_prob:
            for j in range(i * cassette_size, (i + 1) * cassette_size):
                new_dropout_indices.add(j)

    network.nodes[node]["characters"] = new_sample
    network.nodes[node]["dropout"] = list(new_dropout_indices)

    if network.out_degree(node) > 0:
        for i in network.successors(node):
            network.nodes[i]["parent_lifespan"] = network.get_edge_data(
                node, i
            )["weight"]
            mutation_helper_continuous(
                network,
                i,
                basal_rate,
                mutation_prob_map,
                new_sample,
                new_dropout_indices,
                cassette_size,
                epsilon,
                silence_rates,
            )


def add_heritable_dropout(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    heritable_dropout_helper(network, root)


def heritable_dropout_helper(network, node):
    new_sample = network.nodes[node]["characters"].copy()
    for i in network.nodes[node]["dropout"]:
        new_sample[i] = -1
    network.nodes[node]["characters"] = new_sample

    if network.out_degree(node) > 0:
        for i in network.successors(node):
            heritable_dropout_helper(network, i)


def add_stochastic_dropout(network, leaves, dropout_prob, cassette_size):
    for node in leaves:
        new_sample = network.nodes[node]["characters"]
        for i in range(0, len(new_sample) // cassette_size):
            if random.uniform(0, 1) <= dropout_prob:
                for j in range(i * cassette_size, (i + 1) * cassette_size):
                    if new_sample[j] != -1:
                        new_sample[j] = -2
        network.nodes[node]["characters"] = new_sample


def collapse_unifurcations(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    succs = []
    for node in network.successors(root):
        succs.append(node)
    while len(succs) < 2:
        node = succs[0]
        t = network.get_edge_data(root, node)["weight"]
        for i in network.successors(node):
            t_ = network.get_edge_data(node, i)["weight"]
            network.add_edge(root, i, weight=t + t_)
            succs.append(i)
        network.remove_node(node)
        succs.remove(node)
    for node in succs:
        collapse_unifurcations_helper(network, node, root)


def collapse_unifurcations_helper(network, node, parent):
    succs = []
    for i in network.successors(node):
        succs.append(i)
    if len(succs) == 1:
        t = network.get_edge_data(parent, node)["weight"]
        for i in succs:
            t_ = network.get_edge_data(node, i)["weight"]
            network.add_edge(parent, i, weight=t + t_)
            collapse_unifurcations_helper(network, i, parent)
        network.remove_node(node)
    else:
        for i in succs:
            collapse_unifurcations_helper(network, i, node)


########################################################################
def record_heritable_dropouts(network):
    drops = {}
    dist = degree_dist_leaves(network)
    root = [n for n in network if network.in_degree(n) == 0][0]
    record_heritable_helper(network, root, [], dist, drops)
    return drops


def record_heritable_helper(network, node, prev_drops, dist, drops):
    curr_drops = network.nodes[node]["dropout"]
    to_add = list(set(curr_drops) - set(prev_drops))
    if len(to_add) > 0:
        for i in to_add:
            if i in drops:
                drops[i].append(dist[node])
            else:
                drops[i] = [dist[node]]
    if network.out_degree(node) > 0:
        for j in network.successors(node):
            record_heritable_helper(network, j, curr_drops, dist, drops)


def degree_dist_leaves(network):  # returns num of leaf descendants of each node
    root = [n for n in network if network.in_degree(n) == 0][0]
    dist = {}
    leaves_dist_helper(network, root, dist)
    return dist


def leaves_dist_helper(network, node, dist):
    if network.out_degree(node) == 0:
        dist[node] = 1
        return 1
    else:
        total_leaves = 0
        for i in network.successors(node):
            total_leaves += leaves_dist_helper(network, i, dist)
        dist[node] = total_leaves
        return total_leaves


def generate_state_distribution(
    num_characters,
    num_states,
    generating_function,
    equal_for_all_characters=True,
):
    prior_probabilities = {}
    sampled_probabilities = sorted(
        [generating_function() for _ in range(1, num_states + 1)]
    )

    total = np.sum(sampled_probabilities)

    sampled_probabilities = list(
        map(lambda x: x / (1.0 * total), sampled_probabilities)
    )

    prior_probabilities[0] = {}
    for j in range(1, num_states + 1):
        prior_probabilities[0][str(j)] = sampled_probabilities[j - 1]

    if equal_for_all_characters:
        for i in range(range(1, num_characters)):
            prior_probabilities[i] = prior_probabilities[i]
    else:
        for i in range(1, num_characters):
            sampled_probabilities = sorted(
                [generating_function for _ in range(1, num_states + 1)]
            )

            total = np.sum(sampled_probabilities)

            sampled_probabilities = list(
                map(lambda x: x / (1.0 * total), sampled_probabilities)
            )

            prior_probabilities[i] = {}
            for j in range(1, num_states + 1):
                prior_probabilities[i][str(j)] = sampled_probabilities[j - 1]

    return prior_probabilities


############################################################################


class AllLineagesDeadError(Exception):
    """An Exception class for forward simulations, indicating all extant
    lineages had died before the end of the experiment.
    """

    pass


def generate_birth_death(
    birth_waiting_dist,
    death_waiting_dist,
    birth_scale_param,
    death_scale_param,
    fitness_num_dist=None,
    fitness_strength_dist=None,
    num_extant=None,
    experiment_time=None,
):
    unique_id = 0
    tree = nx.DiGraph()
    tree.add_node(unique_id)
    # Keep a list of extant lineages and a dict of lineage specific parameters
    # for each
    current_lineages = []
    current_lineages.append(
        {
            "id": unique_id,
            "birth_scale": birth_scale_param,
            "death_scale": death_scale_param,
            "total_time": 0,
        }
    )

    while len(current_lineages > 1):
        lineage = current_lineages.pop()  # Make sure to pop from front
        birth_waiting_time = birth_waiting_dist(lineage["birth_scale"])
        death_waiting_time = death_waiting_dist(lineage["death_scale"])
        # Take the minimum waiting time to dictate which event happens
        if birth_waiting_time < death_waiting_time:
            # If time is the stopping condition, if the next birth would
            # happen after the time of the experiment, cut the lineage off
            # at the total stopping time and produce no additional daughters
            if experiment_time:
                if (
                    birth_waiting_time + lineage["total_time"]
                    >= experiment_time
                ):
                    parent = list(tree.successors(lineage["id"]))[0]
                    tree.edges[parent][lineage["id"]]["weight"] += (
                        experiment_time - lineage["total_time"]
                    )
            # Determine the number of mutations acquired, and then determine
            # If they are mutations that affect birth or death. Then
            # determine their strength
            num_mutations = fitness_num_dist()
            total_birth_mutation_strength = 1
            total_death_mutation_strength = 1
            for _ in range(num_mutations):
                if np.random.uniform() < 0.5:
                    total_birth_mutation_strength *= fitness_strength_dist()
                else:
                    total_death_mutation_strength *= fitness_strength_dist()
            # Add two daughters with updated total time, and scale parameters
            for i in range(unique_id + 1, unique_id + 3):
                current_lineages.append(
                    {
                        "id": i,
                        "birth_scale": lineage["birth_scale"]
                        * total_birth_mutation_strength,
                        "death_scale": lineage["death_scale"]
                        * total_death_mutation_strength,
                        "total_time": lineage["total_time"]
                        + birth_waiting_time,
                    }
                )
                tree.add_edge(lineage["id"], i, weight=birth_waiting_time)
            unique_id += 2
            # If number of extant lineages is the stopping criterion, at the
            # first instance of having n extant tips, stop the experiment
            # and set the total lineage time for each lineage to be equal to
            # the maximum, to produce ultrametric trees
            if num_extant:
                if len(current_lineages) == num_extant:
                    max_total_time = 0
                    for remaining_lineage in current_lineages:
                        if remaining_lineage["total_time"] > max_total_time:
                            max_total_time = remaining_lineage["total_time"]
                    for remaining_lineage in current_lineages:
                        parent = list(tree.successors(lineage["id"]))[0]
                        tree.edges[parent][lineage["id"]]["weight"] += (
                            max_total_time - lineage["total_time"]
                        )
                    current_lineages = []
        else:
            # If the lineage dies, prune the entire lineage from the tree
            remove_and_prune(lineage["id"], tree)

    if len(tree.nodes) == 0:
        raise AllLineagesDeadError()

    return tree


def estimate_birth_death_rate(t, n, e):
    return (
        1
        / t
        * (
            np.log(
                1 / 2 * n * (1 - e ** 2)
                + 2 * e
                + 1
                / 2
                * (1 - e)
                * np.sqrt(n * (n * e ** 2 - 8 * e + 2 * n * e + n))
            )
            - np.log(2)
        )
    )
