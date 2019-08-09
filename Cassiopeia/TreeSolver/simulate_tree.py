from __future__ import division

import numpy as np
import pandas as pd
import random
from pylab import *
import pickle as pic

import Bio.Phylo as Phylo
import networkx as nx

import sys
import os

import argparse

from Cassiopeia.TreeSolver import *
from Cassiopeia.TreeSolver.lineage_solver import *
from Cassiopeia.TreeSolver.simulation_tools.dataset_generation import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("out_fp", type=str, help="Output file path")
    parser.add_argument('--subsample_percentage', default=0.4, type = float, help="Percentage of cells to sample from final pool")
    parser.add_argument("--depth", "-d", default=10, type = int, help='Depth of Tree to Simulate')
    parser.add_argument('--dropout_rate', "-dr", default=None, help="Dictionary of dropout rates per character")
    parser.add_argument("--num_characters", "-c", default=40, type = int, help='Number of characters to simulate')
    parser.add_argument("--num_states", "-s", default=10, type = int, help="Number of states to simulate")
    parser.add_argument("--mutation_rate", '-m', default=0.025, type = float, help="Mutation rate, assumed to be constant across all characters")
    parser.add_argument("--allele_table", default=None, help="Optional alleletable to provide, where parameters will be estimated from")
    parser.add_argument("--mutation_map", default={}, help="Probabilities of mutating to each state. Given as a nested dictionary.")

    args = parser.parse_args()
    output_file = args.out_fp
    subsample_percentage = args.subsample_percentage 
    depth = args.depth
    number_of_characters = args.num_characters
    number_of_states = args.num_states
    dropout_rates = args.dropout_rate
    mutation_rate = args.mutation_rate
    allele_table = args.allele_table
    prior_probabilities = args.mutation_map

    if allele_table is not None:
        at = pd.read_csv(allele_table, sep='\t')
        piv = pd.pivot_table(at, index=["cellBC"], columns=["intBC"], values="UMI", aggfunc=size)

        if dropout_rates is None:

            dropouts = piv.apply(lambda x: x.isnull().sum() / len(x), axis=0)

        nunique_chars = []
        for n, g in at.groupby(["intBC"]):
        
            nunique_chars.append(len(unique(g["r1"])))
            nunique_chars.append(len(unique(g["r2"])))
            nunique_chars.append(len(unique(g["r3"])))

        number_of_characters = piv.shape[1] * 3 # num char = num intbc * 3
        number_of_states = np.median(nunique_chars)
        
    no_mut_rate = 1 - mutation_rate
    if len(prior_probabilities) == 0:
        for i in range(0, number_of_characters):
            sampled_probabilities = sorted([np.random.negative_binomial(5,.5) for _ in range(1,number_of_states)])
            prior_probabilities[i] = {'0': no_mut_rate}
            total = np.sum(sampled_probabilities)
            for j in range(1, number_of_states):
                prior_probabilities[i][str(j)] = (mutation_rate)*sampled_probabilities[j-1]/(1.0 * total)

        with open("prior_probs.txt", "w") as f:
        
            for i in range(0, number_of_characters):
                f.write(str(i))

                for j in range(1, number_of_states):
                    f.write("\t" + str(prior_probabilities[i][str(j)]))

                f.write("\n")

            f.write("\n")

    if dropout_rates is not None:
        dropouts = pd.read_csv(dropout_rate, sep='\t', index_col = 0) 

    else:
        dropouts = pd.DataFrame(np.full((number_of_characters, 1), 0.1, dtype=float))

    print("Simulating with " + str(number_of_characters) + " Characters and " + str(number_of_states) + " Unique States")
    print("Depth: " + str(depth) + "\nSubsample percentage: " + str(subsample_percentage))
    print("Dropout probabilities:")
    print(dropouts)

    # Generate dropout probabilities
    data_dropout_rates = {}
    j = 0
    for i in range(number_of_characters):
        if allele_table is not None:
            if i != 0 and i % 3 == 0:
                j += 1
            data_dropout_rates[i] = float(dropouts.iloc[j])
        else:
            data_dropout_rates[i] = float(dropouts.iloc[i])

    dropout_prob_map = {i: np.random.choice(list(data_dropout_rates.values())) for i in range(0,number_of_characters)}

    # Generate simulated network
    tree = generate_simulated_full_tree(prior_probabilities, dropout_prob_map, characters=number_of_characters, subsample_percentage=subsample_percentage, depth=depth)
    # tree = Cassiopeia_Tree('simulated', network=true_network)
    pic.dump(tree, open(output_file, "wb"))
