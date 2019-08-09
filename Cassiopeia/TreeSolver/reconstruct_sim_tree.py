from __future__ import division
from __future__ import print_function

import subprocess
import time
from string import ascii_uppercase

import numpy as np
import pandas as pd
import pandascharm as pc
import random
from pylab import *
import pickle as pic
from pathlib import Path

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
import networkx as nx

from skbio import DistanceMatrix
from skbio.tree import nj

import scipy as sp

import sys
import os

import argparse

from Cassiopeia.TreeSolver.lineage_solver import *
from Cassiopeia.TreeSolver.data_pipeline import convert_network_to_newick_format, newick_to_network
from Cassiopeia.TreeSolver.simulation_tools.simulation_utils import *
from Cassiopeia.TreeSolver.utilities import fill_in_tree, tree_collapse
from Cassiopeia.TreeSolver import Cassiopeia_Tree, Node

from numba import jit

import Cassiopeia as sclt

SCLT_PATH = Path(sclt.__path__[0])

@jit(parallel=True)
def compute_distance_mat(cm, C, priors=None):
        
    dm = np.zeros(C * (C-1) // 2, dtype=float)
    k = 0
    for i in range(C-1):
        for j in range(i+1, C):
            
            s1 = cm[i]
            s2 = cm[j]
            
            dm[k] = pairwise_dist(s1, s2, priors)   
            k += 1
    
    return dm

def pairwise_dist(s1, s2, priors=None):
    
    d = 0
    num_present = 0
    for i in range(len(s1)):
        
        if s1[i] == '-' or s2[i] == '-':
            continue
        
        num_present += 1

        if priors:
            if s1[i] == s2[i]:
                d += np.log(priors[i][str(s1[i])])

        if s1[i] != s2[i]:
            if s1[i] == '0' or s2[i] == '0':
                if priors:
                    if s1[i] != '0':
                        d -= np.log(priors[i][str(s1[i])])
                    else:
                        d -= np.log(priors[i][str(s2[i])])
                else:
                    d += 1
            else:
                if priors:
                    d -= (np.log(priors[i][str(s1[i])]) + np.log(priors[i][str(s2[i])]))
                else:
                    d += 2
    if num_present == 0:
        return 0
    
    return d / num_present

def write_leaves_to_charmat(target_nodes, fn):
    """
    Helper function to write TARGET_NODES to a character matrix to conver to multistate;
    needed to run camin-sokal.
    """

    number_of_characters = len(target_nodes[0].char_string.split("|"))
    with open(fn, "w") as f:

        f.write("cellBC")
        for i in range(number_of_characters):
            f.write("\t" + str(i))
        f.write("\n")

        for n in target_nodes:
            charstring, sname = n.char_string, n.name

            f.write(sname)
            chars = charstring.split("|")
            for c in chars:
                f.write("\t" + c)
            f.write("\n")


def unique_alignments(aln):

    new_aln = []
    obs = []
    for a in aln:

        if a.seq in obs:
            continue

        new_aln.append(a)
        obs.append(a.seq)

    return MultipleSeqAlignment(new_aln)

def nx_to_charmat(target_nodes):

    number_of_characters = len(target_nodes[0].split("|"))
    cm = pd.DataFrame(np.zeros((len(target_nodes), number_of_characters)))

    ind = []
    for i in range(len(target_nodes)):
        nr = []
        n = target_nodes[i]
        charstring, sname = n.split("_")
        ind.append("s" + sname)
        chars = charstring.split("|")
        for c in chars:
            nr.append(c)

        cm.iloc[i] = np.array(nr)

    cm.columns = [("r" + str(i)) for i in range(number_of_characters)]
    cm.index = ind

    return cm

def construct_weights(phy, weights_fn, write=True):
    """
    Given some binary phylip infile file path, compute the character-wise log frequencies
    and translate to the phylip scaling (0-Z) for the weights file.
    """

    aln = AlignIO.read(phy, "phylip")

    df = pc.from_bioalignment(aln)

    abund = df.apply(lambda x: len(x[x=="1"]) / len(x), axis=1)

    labund = np.array(list(map(lambda x: float(-1 * np.log2(x)) if x > 1 else x, abund)))
    labund[labund == 0] = labund.min()

    # scale linearly to range for phylip weights
    _min = 0
    _max = 35

    scaled = (_max - _min) / (labund.max() - labund.min()) * (labund - labund.max()) + _max
    scaled = list(map(lambda x: int(x), scaled))

    weights_range = [str(i) for i in range(10)] + [l for l in ascii_uppercase]
    weights_dict = dict(zip(range(36), weights_range))

    scaled = list(map(lambda x: weights_dict[x], scaled))

    if write:
        with open(weights_fn, "w") as f:
            f.write(''.join(scaled))

    return scaled


def main():
    """
    Takes in a character matrix, an algorithm, and an output file and
    returns a tree in newick format.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("netfp", type = str, help="character_matrix")
    parser.add_argument("typ", type=str, help="category of stress test")
    parser.add_argument("-nj", "--neighbor-joining", action="store_true", default=False)
    parser.add_argument("--neighbor_joining_weighted", action='store_true', default=False)
    parser.add_argument("--ilp", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    parser.add_argument("--cutoff", type=int, default=80, help="Cutoff for ILP during Hybrid algorithm")
    parser.add_argument("--time_limit", type=int, default=-1, help="Time limit for ILP convergence")
    parser.add_argument("--iter_limit", type = int, default = -1, help="Max number of iterations for ILP solver")
    parser.add_argument("--greedy", "-g", action="store_true", default=False)
    parser.add_argument("--camin-sokal", "-cs", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False, help="output verbosity")
    parser.add_argument("--mutation_map", type=str, default="")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--no_triplets", action="store_true", default=False)
    parser.add_argument("--max_neighborhood_size", type=str, default=3000)
    parser.add_argument("--out_fp", type=str, default=None, help="optional output file")
    parser.add_argument("--seed", type = int, default = None, help="Random seed for ILP solver")

    args = parser.parse_args()

    netfp = args.netfp
    outfp = args.out_fp
    t = args.typ
    verbose = args.verbose

    cutoff = args.cutoff
    time_limit = args.time_limit
    iter_limit = args.iter_limit
    num_threads = args.num_threads
    max_neighborhood_size = args.max_neighborhood_size
    seed = args.seed

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    score_triplets = (not args.no_triplets)

    prior_probs = None
    if args.mutation_map != "":

        prior_probs = pic.load(open(args.mutation_map, "rb"))

    name = netfp.split("/")[-1]
    stem = '.'.join(name.split(".")[:-1])

    true_network = nx.read_gpickle(netfp)

    if isinstance(true_network, Cassiopeia_Tree):
        true_network = true_network.get_network()

    target_nodes = get_leaves_of_tree(true_network)

    target_nodes_uniq = []
    seen_charstrings = []
    for t in target_nodes:
        if t.char_string not in seen_charstrings:
            seen_charstrings.append(t.char_string)
            target_nodes_uniq.append(t)

    if args.greedy:

        if verbose:
            print('Running Greedy Algorithm on ' + str(len(target_nodes_uniq)) + " Cells")

        reconstructed_network_greedy = solve_lineage_instance(target_nodes_uniq, method="greedy", prior_probabilities=prior_probs)

        net = reconstructed_network_greedy

        #reconstructed_network_greedy = nx.relabel_nodes(reconstructed_network_greedy, string_to_sample)

        if outfp is None:
            outfp = name.replace('true', 'greedy')
        pic.dump(net, open(outfp, "wb"))


    elif args.hybrid:

        if verbose:
            print('Running Hybrid Algorithm on ' + str(len(target_nodes_uniq)) + " Cells")
            print('Parameters: ILP on sets of ' + str(cutoff) + ' cells ' + str(time_limit) + 's to complete optimization')

        reconstructed_network_hybrid = solve_lineage_instance(target_nodes_uniq,  method="hybrid", hybrid_subset_cutoff=cutoff, prior_probabilities=prior_probs, time_limit=time_limit, threads=num_threads, max_neighborhood_size=max_neighborhood_size, seed = seed, num_iter=iter_limit)

        net = reconstructed_network_hybrid

        if outfp is None:
            outfp = name.replace('true', 'hybrid')
        pic.dump(net, open(outfp, "wb"))


    elif args.ilp:

        if verbose:
            print('Running Hybrid Algorithm on ' + str(len(target_nodes_uniq)) + " Cells")
            print('Parameters: ILP on sets of ' + str(cutoff) + ' cells ' + str(time_limit) + 's to complete optimization')

        reconstructed_network_ilp = solve_lineage_instance(target_nodes_uniq, method="ilp", hybrid_subset_cutoff=cutoff, prior_probabilities=prior_probs, 
                                    time_limit=time_limit, max_neighborhood_size = max_neighborhood_size, seed = seed, num_iter=iter_limit)

        net = reconstructed_network_ilp
        # reconstructed_network_ilp = nx.relabel_nodes(reconstructed_network_ilp, string_to_sample)
        if outfp is None:
            outfp = name.replace('true', 'ilp')
        pic.dump(net, open(outfp, 'wb'))


    elif args.neighbor_joining:

        if verbose:
            print("Running Neighbor-Joining on " + str(len(target_nodes_uniq)) + " Unique Cells")


        infile = ''.join(name.split(".")[:-1]) + 'infile.txt'
        fn = ''.join(name.split(".")[:-1]) + 'phylo.txt'
        write_leaves_to_charmat(target_nodes_uniq, fn)

        script = (SCLT_PATH / 'TreeSolver' / 'binarize_multistate_charmat.py')
        cmd =  "python3.6 " + str(script) +  " " + fn + " " + infile + " --relaxed" 
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

        aln = AlignIO.read(infile, "phylip-relaxed")

        aln = unique_alignments(aln)

        t0 = time.time()
        calculator = DistanceCalculator('identity', skip_letters='?')
        constructor = DistanceTreeConstructor(calculator, 'nj')

        tree = constructor.build_tree(aln)

        tree.root_at_midpoint()

        nj_net = Phylo.to_networkx(tree)

        # convert labels to characters for writing to file 
        i = 0
        for n in nj_net:

            if n.name is None:
                n.name = "internal" + str(i)
                i += 1

      
        # convert labels to strings, not Bio.Phylo.Clade objects
        c2str = map(lambda x: x.name, list(nj_net.nodes()))
        c2strdict = dict(zip(list(nj_net.nodes()), c2str))
        nj_net = nx.relabel_nodes(nj_net, c2strdict)

        cm = pd.read_csv(fn, sep='\t', index_col = 0)

        nj_net = fill_in_tree(nj_net, cm)

        nj_net = tree_collapse(nj_net)

        cm_lookup = dict(zip(list(cm.apply(lambda x: "|".join([str(k) for k in x.values]), axis=1)), cm.index.values))

        #rdict = {}
        for n in nj_net:
            #spl = n.split("_")
            #nn = Node('state-node', spl[0].split("|"), is_target = False)
            #if len(spl) > 1:
            #    nn.pid = spl[1] 
            if n.char_string in cm_lookup.keys():
                n.is_target = True
                #nn.name = cm_lookup[spl[0]]
            # rdict[n] = nn

        nj_net = Cassiopeia_Tree('neighbor-joining', network = cs_net)
        if outfp is None:
            outfp = name.replace('true', 'nj')
        pic.dump(nj_net, open(outfp, 'wb'))
        # Phylo.write(tree, out, 'newick')

        os.system("rm " + infile)
        os.system("rm " + fn)

    elif args.neighbor_joining_weighted:

        if verbose:
            print("Running Neighbor-Joining with Weighted Scoring on " + str(len(target_nodes_uniq)) + " Unique Cells")

        target_node_charstrings = np.array([t.get_character_vec() for t in target_nodes_uniq])
        dm = compute_distance_mat(target_node_charstrings, len(target_node_charstrings), priors=prior_probs)
        
        ids = [t.name for t in target_nodes_uniq]
        cm_uniq = pd.DataFrame(target_node_charstrings)
        cm_uniq.index = ids
        dm = sp.spatial.distance.squareform(dm)

        dm = DistanceMatrix(dm, ids)

        newick_str = nj(dm, result_constructor=str)

        tree = newick_to_network(newick_str)

        nj_net = fill_in_tree(tree, cm_uniq)
        nj_net = tree_collapse(nj_net)

        cm_lookup = dict(zip(list(cm_uniq.apply(lambda x: "|".join([str(k) for k in x.values]), axis=1)), cm_uniq.index.values))

        rdict = {}
        for n in nj_net: 
            if n.char_string in cm_lookup:
                n.is_target = True
            else:
                n.is_target = False

        nj_net = Cassiopeia_Tree('neighbor-joining', network = cs_net)
        if outfp is None:
            outfp = name.replace('true', 'nj_weighted')
        pic.dump(nj_net, open(outfp, 'wb'))


    elif args.camin_sokal:

        if verbose:
            print('Running Camin-Sokal Max Parsimony Algorithm on ' + str(len(target_nodes_uniq)) + " Unique Cells")


        samples_to_cells = {}
        indices = []
        for i, n in zip(range(len(target_nodes_uniq)), target_nodes_uniq):
            samples_to_cells["s" + str(i)] = n.name
            indices.append(n.name)
            n.name = str(i)

        infile = ''.join(name.split(".")[:-1]) + '_cs_infile.txt'
        fn = ''.join(name.split(".")[:-1]) + '_cs_phylo.txt'
        weights_fn = ''.join(name.split(".")[:-1]) + "_cs_weights.txt"
        write_leaves_to_charmat(target_nodes_uniq, fn)

        script = (SCLT_PATH / 'TreeSolver' / 'binarize_multistate_charmat.py')
        cmd =  "python3.6 " + str(script) +  " " + fn + " " + infile 
        pi = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(pi.pid, 0)

        weights = construct_weights(infile, weights_fn)

        os.system("touch outfile")
        os.system("touch outtree")

        outfile = stem + 'outfile.txt'
        outtree = stem + 'outtree.txt'
        # run phylip mix with camin-sokal
        responses = "." + stem + ".temp.txt"
        FH = open(responses, 'w')
        current_dir = os.getcwd()
        FH.write(infile + "\n")
        FH.write("F\n" + outfile + "\n")
        FH.write("P\n")
        FH.write("W\n")
        FH.write("Y\n")
        FH.write(weights_fn + "\n")
        FH.write("F\n" + outtree + "\n")
        FH.close()

        t0 = time.time()
        cmd = "~/software/phylip-3.697/exe/mix"
        cmd += " < " + responses + " > screenout1"
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

        consense_outtree = stem + "consenseouttree.txt"
        consense_outfile = stem + "consenseoutfile.txt"

        FH = open(responses, "w")
        FH.write(outtree + "\n")
        FH.write("F\n" + consense_outfile + "\n")
        FH.write("Y\n")
        FH.write("F\n" + consense_outtree + "\n")
        FH.close()

        if verbose:
            print("Computing Consensus Tree, elasped time: " + str(time.time() - t0))

        cmd = "~/software/phylip-3.697/exe/consense"
        cmd += " < " + responses + " > screenout"
        p2 = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p2.pid, 0)

        newick_str = ""
        with open(consense_outtree, "r") as f:
            for l in f:
                l = l.strip()
                newick_str += l

        cm = pd.read_csv(fn, sep='\t', index_col = 0)
        cm.index = indices

        cs_net = newick_to_network(newick_str)

        cs_net = nx.relabel_nodes(cs_net, samples_to_cells)

        cs_net = fill_in_tree(cs_net, cm)

        cs_net = tree_collapse(cs_net)

        cm_lookup = dict(zip(list(cm.apply(lambda x: "|".join([str(k) for k in x.values]), axis=1)), cm.index.values))

        for n in cs_net:
            if n.char_string in cm_lookup.keys():
                n.is_target = True

        cs_net = Cassiopeia_Tree('camin-sokal', network = cs_net)
        if outfp is None:
            outfp = name.replace('true', 'cs')
        pic.dump(cs_net, open(outfp, 'wb'))

        os.system("rm " + outfile)
        os.system("rm " + responses)
        os.system("rm " + outtree)
        os.system("rm " + consense_outfile)
        os.system("rm " + infile)
        os.system("rm " + fn)

    else:

        raise Exception("Please choose an algorithm from the list: greedy, hybrid, ilp, nj, or camin-sokal")

if __name__ == "__main__":
    main()
