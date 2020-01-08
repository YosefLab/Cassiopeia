from __future__ import division
from __future__ import print_function

import pickle as pic

import Bio.Phylo as Phylo
import networkx as nx

import sys
import os

import argparse

sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from simulation_tools.simulation_utils import get_leaves_of_tree
from simulation_tools.validation import check_triplets_correct, tree_collapse
from lineage_solver.solution_evaluation_metrics import cci_score
from data_pipeline import newick_to_network

parser = argparse.ArgumentParser()
parser.add_argument("true_net", type=str)
parser.add_argument("typ", type=str)

args = parser.parse_args()

true_netfp = args.true_net
t = args.typ

name = true_netfp.split("/")[-1]
spl = name.split("_")
param = spl[-3]
run = spl[-1].split(".")[0]

true_network = pic.load(open(true_netfp, "rb"))
target_nodes = get_leaves_of_tree(true_network, clip_identifier=True)

cci_upper = cci_score(target_nodes, bound = "upper")
cci_lower = cci_score(target_nodes, bound = "lower")


print(str(param) + "\t" + str(run) + "\t" + str(cci_lower) + "\t" + str(cci_upper) + '\t' + str(cci_upper - cci_lower) + '\t' + str(t))

