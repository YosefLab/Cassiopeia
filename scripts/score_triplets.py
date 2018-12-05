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
from data_pipeline import newick_to_network

parser = argparse.ArgumentParser()
parser.add_argument("true_net", type=str)
parser.add_argument("r_net", type=str)
parser.add_argument("alg", type=str)
parser.add_argument("typ", type=str)
parser.add_argument("--modified", action="store_true", default=False)

args = parser.parse_args()

true_netfp = args.true_net
reconstructed_fp = args.r_net
alg = args.alg
t = args.typ
modified = args.modified

name = true_netfp.split("/")[-1]
spl = name.split("_")
param = spl[-3]
run = spl[-1].split(".")[0]
#param = "na"

name2 = reconstructed_fp.split("/")[-1]
spl2 = name2.split("_")

ending = spl2[-1].split(".")[-1]

true_network = pic.load(open(true_netfp, "rb"))
target_nodes = get_leaves_of_tree(true_network, clip_identifier=True)
target_nodes_original_network = get_leaves_of_tree(true_network, clip_identifier=False)

if ending == "pkl" or ending == "pickle":

    reconstructed_network = pic.load(open(reconstructed_fp, "rb"))

    nodes = [n for n in reconstructed_network.nodes()]
    encoder = dict(zip(nodes, map(lambda x: x.split("_")[0], nodes)))

    reconstructed_network = nx.relabel_nodes(reconstructed_network, encoder)

else:
    k = map(lambda x: "s" + x.split("_")[-1], target_nodes_original_network)
    s_to_char = dict(zip(k, target_nodes))
    char_to_s = dict(zip(target_nodes, k))

    reconstructed_tree = Phylo.parse(reconstructed_fp, "newick").next()
    reconstructed_tree.rooted = True
    reconstructed_network = Phylo.to_networkx(reconstructed_tree)

    i = 1
    for n in reconstructed_network:
        if n.name is None:
            n.name = "i" + str(i)
            i += 1

    #newick_str = ""
    #with open(reconstructed_fp, "r") as f:
    #    for l in f:
    #        l = l.strip()
    #        newick_str += l

    #reconstructed_tree = newick_to_network(reconstructed_fp)
    #reconstructed_tree = newick_to_network(newick_str)
    #reconstructed_network = tree_collapse(reconstructed_tree)


    # convert labels to strings, not Bio.Phylo.Clade objects
    c2str = map(lambda x: x.name, reconstructed_network.nodes())
    c2strdict = dict(zip(reconstructed_network.nodes(), c2str))
    reconstructed_network  = nx.relabel_nodes(reconstructed_network, c2strdict)

    # convert labels to characters for triplets correct analysis
    reconstructed_network = nx.relabel_nodes(reconstructed_network, s_to_char)
    #reconstructed_network = tree_collapse(reconstructed_network)

tot_tp = 0
if modified:

    correct_class, freqs = check_triplets_correct(true_network, reconstructed_network, dict_return=True)

    num_consid = 0
    for k in correct_class.keys():
        nf = 0
        tp = 0
        if freqs[k] > 20:
            
            num_consid += 1
            tot_tp += correct_class[k] / freqs[k]
        
        #tot_tp += tp / nf

    tot_tp /= num_consid

else:
    tot_tp = check_triplets_correct(true_network, reconstructed_network)

print(str(param) + "\t" + str(run) + "\t" + str(tot_tp) + "\t" + alg  + "\t" + t + "\t" + str(0))
