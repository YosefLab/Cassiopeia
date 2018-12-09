import sys
import networkx as nx 

import pandas as pd
from collections import defaultdict, OrderedDict
import numpy as np

import argparse

from tqdm import tqdm

from SingleCellLineageTracing.TreeSolver import *
from SingleCellLineageTracing.TreeSolver.lineage_solver import *
import random

import pickle as pic


def process_allele_table(cm, old_r = False, mutation_map=None):

    filtered_samples = defaultdict(OrderedDict)
    for sample in cm.index:
        cell = cm.loc[sample, "cellBC"] 
        if old_r:
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_1'] = cm.loc[sample, 'r1.old']
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_2'] = cm.loc[sample, 'r2.old']
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_3'] = cm.loc[sample, 'r3.old']
        else:
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_1'] = cm.loc[sample, 'r1']
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_2'] = cm.loc[sample, 'r2']
	    filtered_samples[cell][cm.loc[sample, 'intBC'] + '_3'] = cm.loc[sample, 'r3']

    samples_as_string = defaultdict(str)
    allele_counter = defaultdict(OrderedDict)

    intbc_uniq = []
    for s in filtered_samples:
        for key in filtered_samples[s]:
            if key not in intbc_uniq:
                intbc_uniq.append(key)

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)
    # for all characters
    for i in tqdm(range(len(list(intbc_uniq))), desc="Processing characters"):

        c = list(intbc_uniq)[i]

        # for all samples, construct a character string
        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:

                state = filtered_samples[sample][c]

                if type(state) != str and np.isnan(state):
                    samples_as_string[sample] += "-|"
                    continue

                if state == "NONE" or "None" in state:
                    samples_as_string[sample] += '0|'
                else:
                    if state in allele_counter[c]:
                        samples_as_string[sample] += str(allele_counter[c][state] + 1) + '|'
                    else:
                        # if this is the first time we're seeing the state for this character,
                        allele_counter[c][state] = len(allele_counter[c]) + 1
                        samples_as_string[sample] += str(allele_counter[c][state] + 1) + '|'

                        # add a new entry to the character's probability map
                        if mutation_map is not None:
                            prob = np.mean(mutation_map.loc[state]['freq'])
                            prior_probs[i][str(len(allele_counter[c]) + 1)] = float(prob)
                            indel_to_charstate[i][str(len(allele_counter[c]) + 1)] = state
            else:
                samples_as_string[sample] += '-|'
    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string, prior_probs, indel_to_charstate

def write_to_charmat(string_sample_values, out_fp):
    
    m = len(string_sample_values[string_sample_values.keys()[0]].split("|"))

    with open(out_fp, "w") as f:

        cols = ["cellBC"] + [("r" + str(i)) for i in range(m)]
        f.write('\t'.join(cols) + "\n")
        
        for k in string_sample_values.keys():

            f.write(k)
            alleles = string_sample_values[k].split("|")

            for a in alleles:
                f.write("\t" + str(a))

            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("at_fp", type = str, help="character_matrix")
    parser.add_argument("out_fp", type=str, help="output file name")
    parser.add_argument("--mutation_map", type=str, default="")
    parser.add_argument("--old_r", action="store_true", default=False)
    
    args = parser.parse_args() 

    at_fp = args.at_fp
    out_fp = args.out_fp
    mutation_map = args.mutation_map
    old_r = args.old_r

    out_stem = ''.join(out_fp.split('.')[:-1])

    at = pd.read_csv(at_fp, sep="\t")

    mut_map = None
    if mutation_map != "":

        mut_map = pd.read_csv(mutation_map, sep = ',', index_col = 0)

    string_sample_values, prior_probs, indel_to_charstate = process_allele_table(at, old_r = old_r, mutation_map=mut_map)

    write_to_charmat(string_sample_values, out_fp)

    # write prior probability dictionary to pickle for convenience
    pic.dump(prior_probs, open(out_stem + "_priorprobs.pkl", "wb"))

    # write indel to character state mapping to pickle 
    pic.dump(indel_to_charstate, open(out_stem + "_indel_character_map.pkl", "wb"))
