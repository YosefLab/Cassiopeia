from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle as pic

import networkx as nx

import sys
import os

import argparse

sys.path.append("/home/mattjones/projects/scLineages/scripts")
sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from data_pipeline import convert_network_to_newick_format, newick_to_network
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
from simulation_tools.dataset_generation import generate_simulated_full_tree
from simulation_tools.simulation_utils import get_leaves_of_tree
from simulation_tools.validation import check_triplets_correct, tree_collapse

def write_leaves_to_charmat(target_nodes, fn, s = False, name_is_charstring = False):
    """
    Helper function to write TARGET_NODES to a character matrix to conver to multistate;
    needed to run camin-sokal.
    """

    number_of_characters = len(target_nodes[0].split("|"))
    with open(fn, "w") as f:

        f.write("cellBC")
        for i in range(number_of_characters):
            f.write("\t" + str(i))
        f.write("\n")

        for n in target_nodes:
            charstring, sname = n.split("_")
            if name_is_charstring:
                sname = charstring
            else:
                if s:
                    sname = "s" + sname
            f.write(sname)
            chars = charstring.split("|")
            for c in chars:
                f.write("\t" + c)
            f.write("\n")



if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("netfp", type=str)
    parser.add_argument("outfp", type=str)
    parser.add_argument("--add_s", action="store_true", default=False)
    parser.add_argument("--name_is_charstring", action="store_true", default=False)

    args = parser.parse_args()

    netfp = args.netfp
    outfp = args.outfp
    add_s = args.add_s
    name_is_charstring = args.name_is_charstring

    g = nx.read_gpickle(netfp)

    target_nodes = get_leaves_of_tree(g, clip_identifier=True)
    target_nodes_original_network = get_leaves_of_tree(g, clip_identifier=False)

    k = map(lambda x: "s" + x.split("_")[-1], target_nodes_original_network)
    s_to_char = dict(zip(k, target_nodes))
    char_to_s = dict(zip(target_nodes, k))

    unique_ii = np.unique(target_nodes, return_index=True)
    target_nodes_uniq = np.array(target_nodes)[unique_ii[1]]
    target_nodes_original_network_uniq = np.array(target_nodes_original_network)[unique_ii[1]]

    write_leaves_to_charmat(target_nodes_original_network_uniq, outfp, s = add_s, name_is_charstring=name_is_charstring)

    
