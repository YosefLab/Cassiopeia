import sys
import networkx as nx 

import pandas as pd
from collections import defaultdict

sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")
from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
import random

import pickle as pic


def process_allele_table(cm):

    filtered_samples = defaultdict(dict)
    for sample in cm.index:
        cell = cm.loc[sample, "cellBC"] 
	filtered_samples[cell][cm.loc[sample, 'intBC'] + '_1'] = cm.loc[sample, 'r1']
	filtered_samples[cell][cm.loc[sample, 'intBC'] + '_2'] = cm.loc[sample, 'r2']
	filtered_samples[cell][cm.loc[sample, 'intBC'] + '_3'] = cm.loc[sample, 'r3']

    samples_as_string = defaultdict(str)
    allele_counter = defaultdict(dict)
    
    intbc_uniq = set()
    for s in filtered_samples:
        for key in filtered_samples[s]:
            intbc_uniq.add(key)

    for c in list(intbc_uniq):

        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:
                if filtered_samples[sample][c] == "NONE":
                    samples_as_string[sample] += '0|'
                else:
                    if filtered_samples[sample][c] in allele_counter[c]:
                        samples_as_string[sample] += str(allele_counter[c][filtered_samples[sample][c]] + 1) + '|'
                    else:
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

    at_fp = sys.argv[1]
    out_fp = sys.argv[2]

    at = pd.read_csv(at_fp, sep="\t")

    string_sample_values = process_allele_table(at)

    write_to_charmat(string_sample_values, out_fp)
