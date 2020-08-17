import sys 
import networkx as nx
import pandas as pd
import numpy as np
import pickle as pic
import random
import scipy.stats as stats

import cassiopeia.TreeSolver.simulation_tools.simulation_utils as sim_utils
import cassiopeia.TreeSolver.simulation_tools.dataset_generation as data_gen
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

import subprocess

from Bio import Phylo as Phylo
from io import StringIO

#import seaborn as sns
import os

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

utils.install_packages('mnormt')
utils.install_packages('TreeSimGM')
utils.install_packages('ape')

def maxDepth(network, node): #tree height
    if network.out_degree(node) == 0:
        return 1
    else:
        depths = []
        for i in network.successors(node):
            depths.append(maxDepth(network, i))
        return max(depths) + 1
    
def all_depths(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    depths = all_depths_helper(network, root)
    return depths, sum(depths)/len(depths)
    
def all_depths_helper(network, node):
    if (network.out_degree(node) == 0):
        return [1]
    else:
        depths = []
        for i in network.successors(node):
            depths.extend(all_depths_helper(network, i))
        return [x + 1 for x in depths]
    
def diameter(network): #tree diameter
    root = [n for n in network if network.in_degree(n) == 0][0]
    if (network.out_degree(root) == 0):  
        return 1
    ans = [-1]
    height_of_tree = maxDepth_help(network, root, ans)  
    return ans[0]

def maxDepth_help(network, node, ans): #tree_height helper for tree diameter
    if network.out_degree(node) == 0:
        ans[0] = max(ans[0], 1)
        return 1
    else:
        depths = []
        for i in network.successors(node):
            depths.append(maxDepth_help(network, i, ans))
        ans[0] = max(ans[0], 1 + sum(depths))  
        return max(depths) + 1

def degree_dist(network): #returns dist of out degrees, and average out degree
    out_degrees = []
    for i in network.nodes:
        out_degrees.append(network.out_degree(i))
    fullness = out_degrees.count(1)/len(out_degrees)
    return out_degrees, sum(out_degrees)/len(out_degrees), fullness

def degree_dist_leaves(network): #returns dist of leaf descendants of each node
    root = [n for n in network if network.in_degree(n) == 0][0]
    dist = []
    leaves_dist_helper(network, root, dist)
    return dist, sum(dist)/len(dist)
    
def leaves_dist_helper(network, node, dist):
    if network.out_degree(node) == 0:
        dist.append(0)
        return 1
    else:
        total_leaves = 0
        for i in network.successors(node):
            total_leaves += leaves_dist_helper(network, i, dist)
        dist.append(total_leaves)
        return(total_leaves)
    
def get_character_matrix(nodes, sampling_proportion = 1):
    
    char_arrays = []
    sel_size = round(sampling_proportion * len(nodes))
    sel_nodes = np.random.choice(nodes, sel_size, replace = False)
    for n in sel_nodes:
        chars = n.char_string.split("_")[0].split("|")
        char_arrays.append(chars)
        
    return pd.DataFrame(char_arrays)

def prune_directed_graph(network):
    root = [a for a in network.nodes()][0]
    if network.out_degree(root) > 0:
        for node in network.successors(root):
            directed_graph_helper(network, node, root)
    return
    
def directed_graph_helper(network, node, parent):
    network.remove_edge(node, parent)
    for i in network.successors(node):
            directed_graph_helper(network, i, node)
    return

def balance_dist(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    height_diffs = []
    max_ = [0]
    balance_dist_helper(network, root, height_diffs, max_)
    return height_diffs, sum(height_diffs)/len(height_diffs), max_[0]

def balance_dist_helper(network, node, ans, max_): #tree height helper for distribution of height differences
    if network.out_degree(node) == 0:
        return 1
    else:
        depths = []
        for i in network.successors(node):
            depths.append(balance_dist_helper(network, i, ans, max_))
        if len(depths) > 1:
            ans.append(max(depths) - min(depths))
            max_[0] = max(max_[0], max(depths) - min(depths))
        return max(depths) + 1
    
def get_colless(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    colless = [0]
    colless_helper(network, root, colless)
    n = len([n for n in network if network.out_degree(n) == 0 and network.in_degree(n) == 1]) 
    return colless[0], (colless[0] - n * np.log(n) - n * (np.euler_gamma - 1 - np.log(2)))/n
    
def colless_helper(network, node, colless):
    if network.out_degree(node) == 0:
        return 1
    else:
        leaves = []
        for i in network.successors(node):
            leaves.append(colless_helper(network, i, colless))
        colless[0] += abs(leaves[0] - leaves[1])
        return sum(leaves)

def states_per_char(cm):
    unique_chars = [0] * cm.shape[1]
#     unique_chars = list(cm.nunique())
#     unique_chars = [x-2 for x in ret]
    for j in range(0, cm.shape[1]):
        unique = set(cm.iloc[:, j])
        if '0' in unique:
            unique.remove('0')
        if '-' in unique:
            unique.remove('-')
        unique_chars[j] = len(unique)
    return unique_chars

def longest_path(node, ans):
    if network.out_degree(node) == 0:
        return
    for i in network.successors(node):
        ans[0] += network.edges()[node, i]['weight']
        longest_path(i, ans)
        break
        
def frange(start, stop, step):
    res, n = start, 1

    while res < stop:
        yield res
        res = start + n * step
        n += 1

def get_net_div(time, n, epsilon):
    ret = []
    test = robjects.r('''
    require(geiger)
    epsilon = ''' + str(epsilon) + '''
    r = bd.ms(time = ''' + str(time) + ''', n = ''' + str(n) + ''', epsilon = epsilon, crown = TRUE)
    lambda = r/(1-epsilon)
    lambda
    ''')
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)

    test = robjects.r('''
    require(geiger)
    epsilon = ''' + str(epsilon) + '''
    r = bd.ms(time = ''' + str(time) + ''', n = ''' + str(n) + ''', epsilon = epsilon, crown = TRUE)
    lambda = r/(1-epsilon)
    mu = lambda * epsilon
    mu
    ''')
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)
    return ret

def get_conf_int(time, div, epsilon, conf):
    ret = []
    test = robjects.r('''
    require(geiger)
    epsilon = ''' + str(epsilon) + '''
    ci = crown.limits(time = ''' + str(time) + ''', ''' + str(div) + ''', epsilon = epsilon, CI = ''' + str(conf) + ''')
    ci[1]
    ''')
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)
    test = robjects.r('''
    require(geiger)
    epsilon = ''' + str(epsilon) + '''
    ci = crown.limits(time = ''' + str(time) + ''', ''' + str(div) + ''', epsilon = epsilon, CI = ''' + str(conf) + ''')
    ci[2]
    ''')
    for item in test.items():
        for i in item:
            if i != None:
                ret.append(i)
    
    return ret

def overlay_mutation_continuous(network, mutation_prob_map, basal_rate, cassette_size, epsilon):
    root = [n for n in network if network.in_degree(n) == 0][0]
    network.nodes[root]['parent_lifespan'] = 0
    
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
    
    mutation_helper_continuous(network, root, basal_rate, mutation_prob_map, root.char_vec, [], cassette_size, epsilon)
            
def mutation_helper_continuous(network, node, basal_rate, mutation_prob_map, curr_mutations, dropout_indices, cassette_size, epsilon):
    new_sample = curr_mutations.copy()
    new_dropout_indices = dropout_indices.copy()
    t = network.nodes[node]['parent_lifespan']
    
    p = 1 - (np.exp(-t * basal_rate))
    
    base_chars = []
    for i in range(0, len(new_sample)):
        if new_sample[i] == '0' and i not in new_dropout_indices:
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
        cass_num = i//cassette_size
        if cass_num in cassettes:
            cassettes[cass_num].append((left, right, i))
        else:
            cassettes[cass_num] = [(left, right, i)]

    for cass_num in cassettes.keys():
        if len(cassettes[cass_num]) > 1:
            time_ranges = []
            for cut in cassettes[cass_num]:
                time_ranges.append(cut)
            time_ranges.sort(key = lambda x: x[0])
            
            seen = set()
            for cut in time_ranges:
                if cut[2] in seen:
                    continue
                for cut2 in time_ranges:
                    if cut2[2] in seen:
                        continue
                    if (cut[1] >= cut2[0]):
                        if cut[2] != cut2[2]:
                            for e in range(min(cut[2], cut2[2]), max(cut[2], cut2[2]) + 1):
                                if e not in new_dropout_indices:
                                    new_dropout_indices.append(e)
                                    seen.add(e)
                                    seen.add(e)
                            break
                seen.add(cut[2])      
        
#     for i in range(0, len(new_sample)//cassette_size):
#         if random.uniform(0, 1) <= 0.027:
#             for j in range(i * cassette_size, (i + 1) * cassette_size):
#                 new_dropout_indices.append(j)
    
    node.char_vec = new_sample
    node.char_string = '|'.join([str(char) for char in new_sample])
    network.nodes[node]['dropout'] = new_dropout_indices
    
    if network.out_degree(node) > 0:
        for i in network.successors(node):
            network.nodes[i]['parent_lifespan'] = network.get_edge_data(node, i)['weight']
            mutation_helper_continuous(network, i, basal_rate, mutation_prob_map, new_sample, new_dropout_indices, cassette_size, epsilon)

def overlay_heritable_dropout(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    h_dropout_helper(network, root)

def h_dropout_helper(network, node):
    new_sample = node.char_vec.copy()
    for i in network.nodes[node]['dropout']:
        new_sample[i] = '-'
    node.char_vec = new_sample
    node.char_string = '|'.join([str(char) for char in new_sample])

    if network.out_degree(node) > 0:
        for i in network.successors(node):
            h_dropout_helper(network, i)
            
def add_stochastic_leaves(leaves, dropout_prob, cassette_size):
    for node in leaves:
        sample = node.char_vec.copy()
        for i in range(0, len(sample)//cassette_size):
            if random.uniform(0, 1) <= dropout_prob:
                for j in range(i * cassette_size, (i + 1) * cassette_size):
                    sample[j] = '-'
        node.char_vec = sample
        node.char_string = '|'.join([str(char) for char in sample])

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

def construct_topology(num_chars, div_rate, death_rate, fitness = None, num_leaves = None, time = None):
    
    if time == None:
        if fitness == None:
            test = robjects.r('''
                require(TreeSimGM)
                require(ape)
                test <- function ()
                {
                    yule <- sim.taxa(1, n=''' + str(num_leaves) + ''', 
                    waitsp="rexp(''' + str(div_rate) + ''')", 
                    waitext="rexp(''' + str(death_rate) + ''')", complete = FALSE)
                    write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                }
                test()
                ''')
        else:
            test = robjects.r('''
                require(TreeSimGM)
                require(ape)
                test <- function ()
                {
                    yule <- sim.taxa(1, n=''' + str(num_leaves) + ''', 
                    waitsp="rexp(''' + str(div_rate) + ''')", 
                    waitext="rexp(''' + str(death_rate) + ''')",
                    shiftsp=list(prob=''' + str(fitness_rate) + ''', strength=“runif(0.5,1.5)”),
                    complete = FALSE)
                    write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                }
                test()
                ''')
        
    elif num_leaves == None:
        try:
            if fitness == None:
                test = robjects.r('''
                    require(TreeSimGM)
                    require(ape)
                    test <- function ()
                    {
                        yule <- sim.age(1, age=''' + str(time) + ''', 
                        waitsp="rexp(''' + str(div_rate) + ''')", 
                        waitext="rexp(''' + str(death_rate) + ''')", complete = FALSE)
                        write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                    }
                    test()
                    ''')
            else:
                test = robjects.r('''
                    require(TreeSimGM)
                    require(ape)
                    test <- function ()
                    {
                        yule <- sim.age(1, age=''' + str(time) + ''', 
                        waitsp="rexp(''' + str(div_rate) + ''')", 
                        waitext="rexp(''' + str(death_rate) + ''')",
                        shiftsp=list(prob=''' + str(fitness_rate) + ''', strength=“runif(0.5,1.5)”),
                        complete = FALSE)
                        write.tree(yule[[1]], file = "", append = FALSE, digits = 10, tree.names = FALSE)
                    }
                    test()
                    ''')
        except Exception as e:
            if e.__class__.__name__ == 'RRuntimeError':          
                print("R Error: likely that lineage fully deceased by the end of experiment, no samples generated")
                return None, []
        
    else:
        print("Please specify either a time length of experiment or a number of cells desired")
        return
    
    for string in test.items():
        string
        
    tree = Phylo.read(StringIO(string[1]), 'newick')
    network = Phylo.to_networkx(tree)
    network = network.to_directed()
    prune_directed_graph(network)
    
    rdict = {}
    i = 0
    for n in network.nodes:
        nn = Node("StateNode" + str(i), [0] * num_chars, pid = i, is_target=False)
        i += 1
        rdict[n] = nn

    network = nx.relabel_nodes(network, rdict)
    
    leaves = [n for n in network if network.out_degree(n) == 0 and network.in_degree(n) == 1] 
    
    state_tree = Cassiopeia_Tree('simulated', network = network)
    return state_tree, leaves