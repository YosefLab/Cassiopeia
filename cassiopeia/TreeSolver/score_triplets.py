from __future__ import division
from __future__ import print_function

import pickle as pic

import Bio.Phylo as Phylo
import networkx as nx

import sys
import os

import argparse

from cassiopeia.TreeSolver.simulation_tools.validation import check_triplets_correct
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

def score_triplets(true_network, reconstructed_network, modified = True, min_size_depth = 20, number_of_trials = 50000):

    if isinstance(true_network, Cassiopeia_Tree):
        stree = true_network
    else:
        stree = Cassiopeia_Tree('simulated', network = true_network)

    if isinstance(reconstructed_network, Cassiopeia_Tree):
        rtree = reconstructed_network
    else:
        rtree = Cassiopeia_Tree('simulated', network = reconstructed_network)

    tot_tp = 0
    if modified:

        correct_class, freqs = check_triplets_correct(stree, rtree,
                                number_of_trials=number_of_trials, dict_return=True)

        num_consid = 0
        for k in correct_class.keys():
            nf = 0
            tp = 0
            if freqs[k] > min_size_depth:

                num_consid += 1
                tot_tp += correct_class[k] / freqs[k]

        tot_tp /= num_consid

    else:

        tot_tp = check_triplets_correct(stree, rtree, number_of_trials = number_of_trials)

    return tot_tp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("true_net", type=str)
    parser.add_argument("r_net", type=str)
    parser.add_argument("alg", type=str)
    parser.add_argument("typ", type=str)
    parser.add_argument("--modified", action="store_true", default=False)
    parser.add_argument('--num_trials', '-n', type=int, default=50000, help="Number of tripelts to sample & compare")
    parser.add_argument("--param", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--depth_thresh", type = int, default=20)

    args = parser.parse_args()

    true_netfp = args.true_net
    reconstructed_fp = args.r_net
    alg = args.alg
    t = args.typ
    modified = args.modified
    num_trials = args.num_trials
    param = args.param
    run = args.run
    d_thresh = args.depth_thresh

    try:
        name = true_netfp.split("/")[-1]
        spl = name.split("_")
        param = spl[-3]
        run = spl[-1].split(".")[0]
    except:
        print("No extra information provided regarding parameters and run; assuming param and run are 0")

    name2 = reconstructed_fp.split("/")[-1]
    spl2 = name2.split("_")

    ending = spl2[-1].split(".")[-1]

    true_network = pic.load(open(true_netfp, "rb"))
    reconstructed_network = pic.load(open(reconstructed_fp, "rb"), encoding = "latin1")

    tot_tp = score_triplets(stree, rtree, number_of_trials=num_trials, modified = modified, min_size_depth = d_thresh)

    print(str(param) + "\t" + str(run) + "\t" + str(tot_tp) + "\t" + alg  + "\t" + t + "\t" + str(0))


if __name__ == "__main__":
    main()
