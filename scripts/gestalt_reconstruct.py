import sys
import networkx as nx 

import pandas as pd
from collections import defaultdict

sys.path.append("/home/mattjones/SingleCellLineageTracing/Alex_Solver")
from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
import random

import pickle as pic


def process_allele_table(cm):

    filtered_samples = defaultdict(dict)
    for sample in cm.index:
        for allele in list(cm.columns):
            filtered_samples[sample][allele] = cm.loc[sample, allele]

    samples_as_string = defaultdict(str)
    allele_counter = defaultdict(dict)

    for c in list(cm.columns):

        for sample in cm.index:

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
                samples_as_string[sample] += '|'

    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string


if __name__ == "__main__":

    at_fp = sys.argv[1]

    at = pd.read_csv(at_fp, sep="\t", index_col=0)

    string_sample_values = process_allele_table(at)

    score = cci_score(string_sample_values.values())

    print "CCI Score", score

    network = solve_lineage_instance(string_sample_values.values(), method="ilp", hybrid_subset_cutoff = 80, time_limit=500)
    
    string_to_sample = dict((string, sample) for sample, string in string_sample_values.items())
    network = nx.relabel_nodes(network, string_to_sample)

    newick = convert_network_to_newick_format(network)

    with open("Z3_charmat.txt", "w") as f:

        for v, k in string_sample_values.items():
            f.write(v + '\t' + k + '\n')

    with open("Z3_newick_ilp.txt", "w") as f:
        f.write(newick)

    pic.dump(network, open("Z3.network.pkl", 'wb'))
    

