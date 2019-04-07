import networkx as nx
from Cassiopeia.TreeSolver import Node
import Cassiopeia.TreeSolver.lineage_solver as ls
import pandas as pd
cm = pd.read_csv("lg2_character_matrix.txt", sep='\t', index_col = 0)
cm_uniq = cm.drop_duplicates(inplace=False)
target_nodes = cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1)
tree = ls.solve_lineage_instance(target_nodes, method= 'greedy')
res = tree.get_network()
print(len([n for n in res if res.in_degree(n) > 1]))
print(len([n for n in res if res.in_degree(n) == 0]))
