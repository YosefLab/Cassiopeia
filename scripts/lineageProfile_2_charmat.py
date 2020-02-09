import sys
import networkx as nx

import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import tqdm

import argparse
sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")
from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
import random

import pickle as pic

def process_lineage_table(lt, mutation_map=None):
    """
    Takes a lineage table and encodes each character state into a unique 
    integer to form a character matrix.

    Equivalent to pd.factorize, but we need to make sure that 'None' strings
    are factorized to 0, and that '-' (missing) data are kept as missing.
    """

    char_mat = lt.copy(deep = True)

    allele_counter = defaultdict(dict)

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)
    # iterate over all characters and factorize the states appropriately
    for i in tqdm(range(len(char_mat.columns)), desc="Factorizing characters"):
        c = char_mat.columns[i]

        for sample in char_mat.index:

            state = char_mat.loc[sample,c]
            if state  == "None":
                char_mat.loc[sample, c] = 0

            elif state != "-":

                if state in allele_counter[c]:
                    char_mat.loc[sample, c] = str(allele_counter[c][char_mat.loc[sample, c]])

                else:

                    if mutation_map is not None:
                        prob = np.mean(mutation_map.loc[state]['freq'])
                        prior_probs[i][str(len(allele_counter[c]) + 1)] = float(prob)
                        indel_to_charstate[i][str(len(allele_counter[c]) + 1)] = state

                    allele_counter[c][state] = len(allele_counter[c]) + 1
                    char_mat.loc[sample, c] = allele_counter[c][state]

    return char_mat, prior_probs, indel_to_charstate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("lt_fp", type=str, help="Lineage Table to be converted to character matrix")
    parser.add_argument("out_fp", type=str, help="Ouput character matrix name")
    parser.add_argument("--mutation_map", type=str, default="", help="3 column mutation matrix, relating indels to their probability")

    args = parser.parse_args()

    lt_fp = args.lt_fp
    out_fp = args.out_fp 
    mutation_map = args.mutation_map

    out_stem = ''.join(out_fp.split('.')[:-1])

    lt = pd.read_csv(lt_fp, sep='\t', index_col = 0)
    
    mut_map = None
    if mutation_map != "":
        mut_map = pd.read_csv(mutation_map, sep=',', index_col=0)

        
    charmat, prior_probs, indel_to_charstate  = process_lineage_table(lt, mutation_map = mut_map)

    # write character matrix
    charmat.to_csv(out_fp, sep='\t')
    
    # write prior probability dictionary to pickle for convenience
    pic.dump(prior_probs, open(out_stem + "_priorprobs.pkl", "wb"))

    # write indel to character state mapping to pickle 
    pic.dump(indel_to_charstate, open(out_stem + "_indel_character_map.pkl", "wb"))
