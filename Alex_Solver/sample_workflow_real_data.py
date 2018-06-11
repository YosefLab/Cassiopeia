import networkx as nx

from data_pipeline import read_and_process_data
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score

string_sample_values = read_and_process_data("ivlt_data/IVLT-ALL.alleleTable2.txt", '2.0')


score = cci_score(string_sample_values.values())

print "CCI Score", score

network = solve_lineage_instance(string_sample_values.values(), method='greedy')

string_to_sample = dict((string, sample) for sample, string in string_sample_values.items())

network = nx.relabel_nodes(network,string_to_sample)