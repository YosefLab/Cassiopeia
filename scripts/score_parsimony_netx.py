from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle as pic

import networkx as nx

import sys
import os

sys.path.append("/home/mattjones/projects/scLineages/scripts")
sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from lineage_solver.solver_utils import get_edge_length 

netfp = sys.argv[1]
charfp = sys.argv[2]
alg = sys.argv[3]
t = sys.argv[4]

name = netfp.split("/")[-1]
spl = name.split("_")
param = spl[-3]
run = spl[-1].split(".")[0]

G = pic.load(open(netfp, "rb"))

cm = pd.read_csv(charfp, sep='\t', index_col=0)

sample_to_charstring = {}
target_nodes = cm.astype(str).apply(lambda x: '|'.join(x), axis=1)
for sample in target_nodes.index:
    sample_to_charstring[sample] = target_nodes.loc[sample]

G = nx.relabel_nodes(G, sample_to_charstring)

score = 0
for e in G.edges():
    score += get_edge_length(e[0], e[1])

print(str(param) + "\t" + str(run) + "\t" + str(score) + "\t" + alg + "\t" + str(t))
