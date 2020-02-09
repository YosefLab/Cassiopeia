from __future__ import division

import subprocess
import numpy as np
import pandas as pd
import random
from pylab import *
import pickle as pic
from pathlib import Path

import argparse
from tqdm import tqdm

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import (
    DistanceCalculator,
    ParsimonyScorer,
    DistanceTreeConstructor,
)
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from skbio import DistanceMatrix
from skbio.tree import nj
from numba import jit
import scipy as sp

import networkx as nx

import sys
import os

from cassiopeia.TreeSolver.lineage_solver import *
from cassiopeia.TreeSolver.simulation_tools import *
from cassiopeia.TreeSolver.utilities import (
    fill_in_tree,
    tree_collapse,
    convert_network_to_newick_format,
)
from cassiopeia.TreeSolver import *
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from cassiopeia.TreeSolver.alternative_algorithms import (
    run_nj_weighted,
    run_nj_naive,
    run_camin_sokal,
)

import cassiopeia as sclt

SCLT_PATH = Path(sclt.__path__[0])


def read_mutation_map(mmap):
    """
    Parse file describing the likelihood of state transtions per character.

    Currently, we're just storing the mutation map as a pickle file, so read in with pickle.
    """

    mut_map = pic.load(open(mmap, "rb"))

    return mut_map


def main():
    """
    Takes in a character matrix, an algorithm, and an output file and 
    returns a tree in newick format.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("char_fp", type=str, help="character_matrix")
    parser.add_argument("out_fp", type=str, help="output file name")
    parser.add_argument("-nj", "--neighbor-joining", action="store_true", default=False)
    parser.add_argument(
        "--neighbor_joining_weighted", action="store_true", default=False
    )
    parser.add_argument("--ilp", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    parser.add_argument(
        "--cutoff", type=int, default=80, help="Cutoff for ILP during Hybrid algorithm"
    )
    parser.add_argument(
        "--hybrid_lca_mode",
        action="store_true",
        help="Use LCA distances to transition in hybrid mode, instead of number of cells",
    )
    parser.add_argument(
        "--time_limit", type=int, default=1500, help="Time limit for ILP convergence"
    )
    parser.add_argument("--greedy", "-g", action="store_true", default=False)
    parser.add_argument("--camin-sokal", "-cs", action="store_true", default=False)
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="output verbosity"
    )
    parser.add_argument("--mutation_map", type=str, default="")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--max_neighborhood_size", type=int, default=10000)
    parser.add_argument("--weighted_ilp", "-w", action="store_true", default=False)
    parser.add_argument("--greedy_min_allele_rep", type=float, default=1.0)
    parser.add_argument("--fuzzy_greedy", action="store_true", default=False)
    parser.add_argument("--multinomial_greedy", action="store_true", default=False)
    parser.add_argument("--num_neighbors", default=10)
    parser.add_argument("--num_alternative_solutions", default=100, type=int)
    parser.add_argument("--greedy_missing_data_mode", default="lookahead", type=str)
    parser.add_argument("--greedy_lookahead_depth", default=3, type=int)

    args = parser.parse_args()

    char_fp = args.char_fp
    out_fp = args.out_fp
    verbose = args.verbose

    lca_mode = args.hybrid_lca_mode
    if lca_mode:
        lca_cutoff = args.cutoff
        cell_cutoff = None
    else:
        cell_cutoff = args.cutoff
        lca_cutoff = None
    time_limit = args.time_limit
    num_threads = args.num_threads

    n_neighbors = args.num_neighbors
    num_alt_soln = args.num_alternative_solutions

    max_neighborhood_size = args.max_neighborhood_size

    missing_data_mode = args.greedy_missing_data_mode
    lookahead_depth = args.greedy_lookahead_depth
    if missing_data_mode not in ["knn", "lookahead", "avg", "modified_avg"]:
        raise Exception("Greedy missing data mode not recognized")

    stem = "".join(char_fp.split(".")[:-1])

    cm = pd.read_csv(char_fp, sep="\t", index_col=0, dtype=str)

    cm_uniq = cm.drop_duplicates(inplace=False)

    cm_lookup = list(cm.apply(lambda x: "|".join(x.values), axis=1))
    newick = ""

    prior_probs = None
    if args.mutation_map != "":

        prior_probs = read_mutation_map(args.mutation_map)

    weighted_ilp = args.weighted_ilp
    if prior_probs is None and weighted_ilp:
        raise Exception(
            "If you'd like to use weighted ILP reconstructions, you need to provide a mutation map (i.e. prior probabilities)"
        )

    greedy_min_allele_rep = args.greedy_min_allele_rep
    fuzzy = args.fuzzy_greedy
    probabilistic = args.multinomial_greedy

    if args.greedy:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print("Read in " + str(cm.shape[0]) + " Cells")
            print(
                "Running Greedy Algorithm on "
                + str(len(target_nodes))
                + " Unique States"
            )

        reconstructed_network_greedy, potential_graph_sizes = solve_lineage_instance(
            target_nodes,
            method="greedy",
            prior_probabilities=prior_probs,
            greedy_minimum_allele_rep=greedy_min_allele_rep,
            fuzzy=fuzzy,
            probabilistic=probabilistic,
            n_neighbors=n_neighbors,
            missing_data_mode=missing_data_mode,
            lookahead_depth=lookahead_depth,
        )

        net = reconstructed_network_greedy.get_network()

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_greedy, open(out_stem + ".pkl", "wb"))

        newick = reconstructed_network_greedy.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

        root = [n for n in net if net.in_degree(n) == 0][0]
        # score parsimony
        score = 0
        for e in nx.dfs_edges(net, source=root):
            score += e[0].get_mut_length(e[1])

        print("Parsimony: " + str(score))

    elif args.hybrid:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print("Running Hybrid Algorithm on " + str(len(target_nodes)) + " Cells")
            if lca_mode:
                print(
                    "Parameters: ILP on sets of cells with a maximum LCA distance of "
                    + str(lca_cutoff)
                    + " with "
                    + str(time_limit)
                    + "s to complete optimization"
                )
            else:
                print(
                    "Parameters: ILP on sets of "
                    + str(cell_cutoff)
                    + " cells with "
                    + str(time_limit)
                    + "s to complete optimization"
                )

        # string_to_sample = dict(zip(target_nodes, cm_uniq.index))

        # target_nodes = list(map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index))

        print("running algorithm...")
        reconstructed_network_hybrid, potential_graph_sizes = solve_lineage_instance(
            target_nodes,
            method="hybrid",
            hybrid_cell_cutoff=cell_cutoff,
            hybrid_lca_cutoff=lca_cutoff,
            prior_probabilities=prior_probs,
            time_limit=time_limit,
            threads=num_threads,
            max_neighborhood_size=max_neighborhood_size,
            weighted_ilp=weighted_ilp,
            greedy_minimum_allele_rep=greedy_min_allele_rep,
            fuzzy=fuzzy,
            probabilistic=probabilistic,
            n_neighbors=n_neighbors,
            maximum_alt_solutions=num_alt_soln,
            missing_data_mode=missing_data_mode,
            lookahead_depth=lookahead_depth,
        )

        net = reconstructed_network_hybrid.get_network()

        if verbose:
            print("Writing the tree to output...")

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_hybrid, open(out_stem + ".pkl", "wb"))

        newick = reconstructed_network_hybrid.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

        ## plot out diagnostic potential graph sizes
        h = plt.figure(figsize=(10, 10))
        for i in range(len(potential_graph_sizes)):
            try:
                x, y = (
                    [k for k in potential_graph_sizes[i].keys()],
                    [
                        potential_graph_sizes[i][k]
                        for k in potential_graph_sizes[i].keys()
                    ],
                )
                plt.plot(x, y)
            except:
                continue
        # plt.xlim(0, int(cutoff))
        plt.xlabel("LCA Distance")
        plt.ylabel("Size of Potential Graph")
        plt.savefig(out_stem + "_potentialgraphsizes.pdf")

        # score parsimony
        score = 0
        for e in net.edges():
            score += e[0].get_mut_length(e[1])

        print("Parsimony: " + str(score))

    elif args.ilp:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print(
                "Running ILP Algorithm on " + str(len(target_nodes)) + " Unique Cells"
            )
            print(
                "Paramters: ILP allowed "
                + str(time_limit)
                + "s to complete optimization"
            )

        reconstructed_network_ilp, potential_graph_sizes = solve_lineage_instance(
            target_nodes,
            method="ilp",
            prior_probabilities=prior_probs,
            time_limit=time_limit,
            max_neighborhood_size=max_neighborhood_size,
            weighted_ilp=weighted_ilp,
            maximum_alt_solutions=num_alt_soln,
        )

        net = reconstructed_network_ilp.get_network()

        root = [n for n in net if net.in_degree(n) == 0][0]

        # score parsimony
        score = 0
        for e in nx.dfs_edges(net, source=root):
            score += e[0].get_mut_length(e[1])

        print("Parsimony: " + str(score))

        newick = reconstructed_network_ilp.get_newick()

        if verbose:
            print("Writing the tree to output...")

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_ilp, open(out_stem + ".pkl", "wb"))

        with open(out_fp, "w") as f:
            f.write(newick)

        h = plt.figure(figsize=(10, 10))
        for i in range(len(potential_graph_sizes)):
            try:
                x, y = (
                    [k for k in potential_graph_sizes[i].keys()],
                    [
                        potential_graph_sizes[i][k]
                        for k in potential_graph_sizes[i].keys()
                    ],
                )
                plt.plot(x, y)
            except:
                continue
        # plt.xlim(0, int(cutoff))
        plt.xlabel("LCA Distance")
        plt.ylabel("Size of Potential Graph")
        plt.savefig(out_stem + "_potentialgraphsizes.pdf")

    elif args.neighbor_joining:

        out_stem = "".join(out_fp.split(".")[:-1])

        ret_tree = run_nj_naive(cm_uniq, stem, verbose)

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb"))

        newick = ret_tree.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

    elif args.neighbor_joining_weighted:

        out_stem = "".join(out_fp.split(".")[:-1])
        ret_tree = run_nj_weighted(cm_uniq, prior_probs, verbose)

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb"))

        newick = ret_tree.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

    elif args.camin_sokal:

        out_stem = "".join(out_fp.split(".")[:-1])

        ret_tree = run_camin_sokal(cm_uniq, stem, verbose)

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb"))

        newick = convert_network_to_newick_format(ret_tree.get_network())
        # newick = ret_tree.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

    elif alg == "--max-likelihood" or alg == "-ml":

        # cells = cm.index
        # samples = [("s" + str(i)) for i in range(len(cells))]
        # samples_to_cells = dict(zip(samples, cells))

        # cm.index = list(range(len(cells)))

        if verbose:
            print("Running Maximum Likelihood on " + str(cm.shape[0]) + " Unique Cells")

        infile = stem + "infile.txt"
        fn = stem + "phylo.txt"

        cm.to_csv(fn, sep="\t")

        script = SCLT_PATH / "TreeSolver" / "binarize_multistate_charmat.py"
        cmd = "python3.6 " + str(script) + " " + fn + " " + infile + " --relaxed"
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

        os.system("/home/mattjones/software/FastTreeMP < " + infile + " > " + out_fp)

        tree = Phylo.parse(out_fp, "newick").next()

        ml_net = Phylo.to_networkx(tree)

        i = 0
        for n in ml_net:
            if n.name is None:
                n.name = "internal" + str(i)
                i += 1

        c2str = map(lambda x: str(x), ml_net.nodes())
        c2strdict = dict(zip(ml_net.nodes(), c2str))
        ml_net = nx.relabel_nodes(ml_net, c2strdict)

        out_stem = "".join(out_fp.split(".")[:-1])

        pic.dump(ml_net, open(out_stem + ".pkl", "wb"))

        os.system("rm " + infile)
        os.system("rm " + fn)

    else:

        raise Exception(
            "Please choose an algorithm from the list: greedy, hybrid, ilp, nj, max-likelihood, or camin-sokal"
        )


if __name__ == "__main__":
    main()
