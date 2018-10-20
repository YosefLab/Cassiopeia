import sys
import networkx as nx 

import pandas as pd
from collections import defaultdict

import argparse

sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")
from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
import random

import pickle as pic


def process_allele_table(cm, old_r = False, mutation_map=None):

    filtered_samples = defaultdict(dict)
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
    allele_counter = defaultdict(dict)
    
    intbc_uniq = set()
    for s in filtered_samples:
        for key in filtered_samples[s]:
            intbc_uniq.add(key)

    prior_probs = {}
    # for all characters
    for c in list(intbc_uniq):
        # for all samples, construct a character string
        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:
                if filtered_samples[sample][c] == "NONE":
                    samples_as_string[sample] += '0|'
                else:
                    if filtered_samples[sample][c] in allele_counter[c]:
                        samples_as_string[sample] += str(allele_counter[c][filtered_samples[sample][c]] + 1) + '|'
                    else:
                        # if this is the first time we're seeing the state for this character,
                        # add a new entry to the character's probability map
                        if mutation_map is not None:
                            prior_probs[len(allele_counter[c]) + 1] = mutation_map[filtered_samples[sample][c]][2]
                        allele_counter[c][filtered_samples[sample][c]] = len(allele_counter[c]) + 1
                        samples_as_string[sample] += str(allele_counter[c][filtered_samples[sample][c]] + 1) + '|'
            else:
                samples_as_string[sample] += '-|'

    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string

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

    at = pd.read_csv(at_fp, sep="\t")

    mut_map = None
    if mutation_map != "":

        mut_map = pd.read_csv(mutation_map, sep = ',', index_col = 0)

    string_sample_values = process_allele_table(at, old_r = old_r, mutation_map=mut_map)

    write_to_charmat(string_sample_values, out_fp)
