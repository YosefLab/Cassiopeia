import sys 
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pic
import random

import cassiopeia.TreeSolver.simulation_tools.simulation_utils as sim_utils
import cassiopeia.TreeSolver.simulation_tools.dataset_generation as data_gen
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

def get_character_matrix(nodes):
    
    char_arrays = []
    for n in nodes:
        chars = n.char_string.split("_")[0].split("|")
        char_arrays.append(chars)
        
    return pd.DataFrame(char_arrays)

def compute_priors(C, S, p, mean=0.01, disp=0.1, skew_factor = 0.05, num_skew=1, empirical = np.array([]), mixture = 0):
    
    sp = {}
    prior_probabilities = {}
    for i in range(0, C):
        if len(empirical) > 0:
            sampled_probabilities = sorted(empirical)
        else:
            sampled_probabilities = sorted([np.random.negative_binomial(mean,disp) for _ in range(1,S+1)])
        s = C % num_skew
        mut_rate = p * (1 + num_skew * skew_factor)
        prior_probabilities[i] = {'0': (1-mut_rate)}
        total = np.sum(sampled_probabilities)

        sampled_probabilities = list(map(lambda x: x / (1.0 * total), sampled_probabilities))
        
        if mixture > 0: 
            for s in range(len(sampled_probabilities)):
                if np.random.uniform() <= mixture:
                    sampled_probabilities[s] = np.random.uniform()
            
            sp[i] = sampled_probabilities 
            total = np.sum(sampled_probabilities)
            sampled_probabilities = list(map(lambda x: x / (1.0 * total), sampled_probabilities))
            
            
        for j in range(1, S+1):
            prior_probabilities[i][str(j)] = (mut_rate)*sampled_probabilities[j-1]

    return prior_probabilities, sp

def count_all_dropouts_leaves(leaves):
    count = 0
    for node in leaves:
        sample = node.get_character_string().split('|')
        for i in sample:
            if (i == '-' or i == '*'):
                count += 1
    return count

def overlay_mutation(network, mutation_prob_map, basal_rate, cassette_size):
    root = [n for n in network if network.in_degree(n) == 0][0]
    mutation_cache = {}
    
    for i in mutation_prob_map: #edit the mutation map to only include the probs 
                                #of mutating to each state, given that character is chosen to mutate
        sum = 0
        mutation_prob_map[i].pop('0', None)
        for j in mutation_prob_map[i]:
            sum += mutation_prob_map[i][j]
        new_probs = {}
        for j in mutation_prob_map[i]:
            new_probs[j] = mutation_prob_map[i][j]/sum
        mutation_prob_map[i] = new_probs
    
    mutation_prob_map['basal_mut_rate'] = basal_rate
    
    mutation_helper(network, root, mutation_prob_map, mutation_cache, root.char_vec, [], cassette_size)

def mutation_helper(network, node, mutation_prob_map, mutation_cache, curr_mutations, dropout_indices, cassette_size):
    new_sample = curr_mutations.copy()
    new_dropout_indices = dropout_indices.copy()
    t = network.nodes[node]['lifespan']
    if t == 0:
        return
    mut_rate = mutation_prob_map['basal_mut_rate']
    p = 0
    
    if len(mutation_cache) == 0:
        mutation_cache[1] = mut_rate
    
    if t in mutation_cache:
        p = mutation_cache[t]
    else:
        t_p = max(mutation_cache.keys())
        p = mutation_cache[t_p]
        for t_temp in range(t_p + 1, t + 1):
            p += mut_rate * (1 - mut_rate) ** (t_temp - 1)
            mutation_cache[t_temp] = p
            
    base_chars = []
    for i in range(0, len(new_sample)):
        if new_sample[i] == '0' and i not in new_dropout_indices:
            base_chars.append(i)
    
    times = {}
    draws = np.random.binomial(len(base_chars), p)
    chosen_ind = np.random.choice(base_chars, draws)
    for i in chosen_ind:
        values, probabilities = zip(*mutation_prob_map[i].items())
        new_character = np.random.choice(values, p=probabilities)
        new_sample[i] = new_character
        time = np.random.choice(range(1, t + 1))
#         for ti in range(1, t + 1):
#             if ti in times:
#                 times[ti].append(i)
#             else:
#                 times[ti] = [i]
        if time in times:
            times[time].append(i)
        else:
            times[time] = [i]
    
    for time in sorted(times.keys()):
        if len(times[time]) > 1:
            not_dropped = []
            for i in times[time]:
                if i not in new_dropout_indices:
                    not_dropped.append(i)
            for c in range(0, (len(new_sample)//cassette_size)):
                cass_indices = []
                for i in not_dropped:
                    if (i >= c * cassette_size and i < (c + 1) * cassette_size):
                        cass_indices.append(i)
                if len(cass_indices) > 1:
                    for e in range(min(cass_indices), max(cass_indices) + 1):
                        new_dropout_indices.append(e)
             
    node.char_vec = new_sample
    node.char_string = '|'.join([str(char) for char in new_sample])
    network.nodes[node]['dropout'] = new_dropout_indices

    if network.out_degree(node) > 0:
        for i in network.successors(node):
            mutation_helper(network, i, mutation_prob_map, mutation_cache, new_sample, new_dropout_indices, cassette_size)

def overlay_heritable_dropout(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    h_dropout_helper(network, root)

def h_dropout_helper(network, node):
    new_sample = node.char_vec
    for i in network.nodes[node]['dropout']:
        new_sample[i] = '-'
    node.char_vec = new_sample
    node.char_string = '|'.join([str(char) for char in new_sample])

    if network.out_degree(node) > 0:
        for i in network.successors(node):
            h_dropout_helper(network, i)

# def mutation_helper_old(network, node, mutation_prob_map, mutation_cache, curr_mutations):
#     new_sample = curr_mutations
#     t = network.nodes[node]['lifespan']
#     for i in range(0, len(new_sample)):
#         if new_sample[i] == '0':
#             values, probabilities = zip(*mutation_prob_map[i].items())
#             if t in mutation_cache[i]:
#                 new_probs = mutation_cache[i][t]
#             else:
#                 new_probs = []
#                 t_p = 0
#                 if len(mutation_cache) == 0:
#                     t_p = 1
#                     new_probs = probabilities
#                 else:
#                     t_p = max(mutation_cache[i])
#                     new_probs = mutation_cache[i][t_p]
#                 for t_temp in range(t_p, t + 1):
#                     new_probs[0] *= probabilities[0]
#                     for p in range(1, len(new_probs)):
#                         new_probs += probabilities[0] ** (t_temp - 1) * probabilities[p]
#                     mutation_cache[i][t_temp] = new_probs
#             new_character = np.random.choice(values, p=new_probs)
#             new_sample[i] = new_character
#     node.char_vec = new_sample
#     node.char_string = '|'.join([str(char) for char in new_sample])
    
#     if network.out_degree(node) > 0:
#         for i in network.successors(node):
#             mutation_helper(network, i, mutation_prob_map, mutation_cache, new_sample)

def phylo_forward_pass(
    cassette_size = 3,
    cassette_number = 10,
    timesteps = 100, 
    min_division_rate = 0.076,
    #U = lambda: np.random.exponential(1, 1),
    fitness_rate = 0.000,
    epsilon = 0.001,
    cell_death = 0.001
):
    
    characters = cassette_size * cassette_number
    
#     division_rate = min_division_rate + np.random.exponential(1, 1) * (1 - min_division_rate) # probability that cell will double per time-step
    division_rate = min_division_rate
    
    network = nx.DiGraph()
    current_cells = [[['0' for _ in range(0, characters)], '0']]
    
    network.add_node(sim_utils.node_to_string(current_cells[0]))
    network.nodes[sim_utils.node_to_string(current_cells[0])]['fitness'] = division_rate
    network.nodes[sim_utils.node_to_string(current_cells[0])]['lifespan'] = 0
    uniq = 1
    
    for t in range(0, timesteps):
        temp_current_cells = []
        if len(current_cells) == 0:
            # print("all cells dead, terminating")
            break
        # current_fitnesses = [network.nodes[sim_utils.node_to_string(n)]['fitness'] for n in current_cells]
        # norm = np.max(current_fitnesses)
        
        for node in current_cells:
            fitness = network.nodes[sim_utils.node_to_string(node)]['fitness']
            network.nodes[sim_utils.node_to_string(node)]['lifespan'] += 1
            
            if np.random.random() >= cell_death:
                
                if np.random.random() <= fitness_rate: #cell gains a fitness mutation
#                     if t == (timesteps - 1):
#                         network.node[node]['fitness'] = fitness
#                     else:
#                     s = max(1e-20, U()[0])
                    if np.random.random() <= 0.5:
                        fitness = fitness + epsilon
                    else:
                        fitness = fitness - epsilon
                    network.nodes[sim_utils.node_to_string(node)]['fitness'] = fitness
                
                if np.random.random() <= fitness and t != (timesteps - 1): #cell divides
                    for _ in range(0,2):
                        parent_fitness = network.nodes[sim_utils.node_to_string(node)]['fitness']
                        temp_current_cells.append([node[0], str(uniq)])
                        network.add_edge(sim_utils.node_to_string(node), sim_utils.node_to_string([node[0], str(uniq)]))
                        network.nodes[sim_utils.node_to_string([node[0], str(uniq)])]['fitness'] = parent_fitness
                        network.nodes[sim_utils.node_to_string([node[0], str(uniq)])]['lifespan'] = 0
                        uniq += 1
                else: #cell does not divide
                    temp_current_cells.append(node)
                                    
            else: #if cell dies
                curr_parent = sim_utils.node_to_string(node)
                while network.out_degree(curr_parent) < 1 and network.in_degree(curr_parent) > 0:
                    next_parent = list(network.predecessors(curr_parent))[0]
                    network.remove_node(curr_parent)
                    curr_parent = next_parent
                
        current_cells = temp_current_cells
#         print("timestep:" + str(t))
#         print("size:" + str(len(current_cells)))
        
    rdict = {}
    i = 0
    for n in network.nodes:
        nn = Node("StateNode" + str(i), n.split("_")[0].split("|"), pid = n.split("_")[1], is_target=False)
        i += 1
        rdict[n] = nn

    network = nx.relabel_nodes(network, rdict)
    
#     source = [x for x in network.nodes() if network.in_degree(x)==0][0]

#     max_depth = max(nx.shortest_path_length(network,source,node) for node in network.nodes())
#     shortest_paths = nx.shortest_path_length(network,source)

#     leaves = [x for x in network.nodes() if network.out_degree(x)==0 and network.in_degree(x) == 1 and shortest_paths[x] == max_depth]

    leaves = [n for n in network if network.out_degree(n) == 0 and network.in_degree(n) == 1] 
    
    state_tree = Cassiopeia_Tree('simulated', network = network)
    return state_tree, leaves

