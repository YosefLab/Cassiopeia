import networkx as nx

from data_pipeline import read_and_process_data, convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score

string_sample_values = read_and_process_data("ivlt_data/IVLT-2B_ALL.alleleTable.txt", '7.0')


score = cci_score(string_sample_values.values())

print("CCI Score" + str(score))

network = solve_lineage_instance(string_sample_values.values(), method='greedy')

string_to_sample = dict((string, sample) for sample, string in string_sample_values.items())

network = nx.relabel_nodes(network,string_to_sample)

print(convert_network_to_newick_format(network))
