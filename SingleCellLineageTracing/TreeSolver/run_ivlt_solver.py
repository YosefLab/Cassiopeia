import networkx as nx
import sys

from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
import random

target_lg = sys.argv[1]
file_fp = sys.argv[2]

string_sample_values = read_and_process_data(file_fp, str(target_lg))
print(len(string_sample_values.keys()))
#ds = random.sample(string_sample_values.items(), 3000)
#ds = dict(ds)

network = solve_lineage_instance(string_sample_values.values(), method="hybrid", hybrid_subset_cutoff = 80, time_limit=500)

string_to_sample = dict((string, sample) for sample, string in string_sample_values.items())
network = nx.relabel_nodes(network, string_to_sample)

newick = convert_network_to_newick_format(network)

with open("lg" + str(target_lg) + "_charmat.txt", "w") as f:
    
    for v, k in string_sample_values.items():
        f.write(v + "\t" + k + "\n")


with open("lg" + str(target_lg) + "_newick_hybrid.ALL.txt", "w") as f:
    f.write(newick + "\t")


