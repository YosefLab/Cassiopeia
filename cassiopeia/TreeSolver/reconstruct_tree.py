from __future__ import division

import subprocess
from string import ascii_uppercase

import numpy as np
import pandas as pd
import pandascharm as pc
import random
from pylab import *
import pickle as pic
from pathlib import Path

import argparse
from tqdm import tqdm

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer, DistanceTreeConstructor
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
from cassiopeia.TreeSolver.utilities import fill_in_tree, tree_collapse2
from cassiopeia.TreeSolver import *
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
import cassiopeia.TreeSolver.data_pipeline as dp

import cassiopeia as sclt

SCLT_PATH = Path(sclt.__path__[0])

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

def pairwise_dist(s1, s2, priors=None):
    
    d = 0
    num_present = 0
    for i in range(len(s1)):
        
        if s1[i] == '-' or s2[i] == '-':
            continue
        
        num_present += 1

        if priors:
            if s1[i] == s2[i] and (s1[i] != '0'):
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

@jit(parallel=True)
def compute_distance_mat(cm, C, priors=None):
        
    dm = np.zeros(C * (C-1) // 2, dtype=float)
    k = 0
    for i in tqdm(range(C-1), desc = 'solving distance matrix'):
        for j in range(i+1, C):
            
            s1 = cm[i,:]
            s2 = cm[j,:]
                
            dm[k] = pairwise_dist(s1, s2, priors=priors)   
            k += 1
    
    return dm

def read_mutation_map(mmap):
    """
    Parse file describing the likelihood of state transtions per character.

    Currently, we're just storing the mutation map as a pickle file, so read in with pickle.
    """
    
    mut_map = pic.load(open(mmap, "rb"))

    return mut_map

def construct_weights(phy, weights_fn, write=True):
    """
    Given some binary phylip infile file path, compute the character-wise log frequencies
    and translate to the phylip scaling (0-Z) for the weights file. 
    """

    aln = AlignIO.read(phy, "phylip")

    df = pc.from_bioalignment(aln)

    abund = df.apply(lambda x: len(x[x=="1"]) / len(x), axis=1)
    
    labund = np.array(list(map(lambda x: float(-1 * np.log2(x)) if x > 1 else x, abund)))
    labund[labund == 0] = np.min(labund)

    # scale linearly to range for phylip weights
    _min = 0
    _max = 35

    scaled = (_max - _min) / (np.max(labund) - np.min(labund)) * (labund - np.max(labund))+ _max
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
    parser.add_argument("char_fp", type = str, help="character_matrix")
    parser.add_argument("out_fp", type=str, help="output file name")
    parser.add_argument("-nj", "--neighbor-joining", action="store_true", default=False)
    parser.add_argument("--neighbor_joining_weighted", action='store_true', default=False)
    parser.add_argument("--ilp", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    parser.add_argument("--cutoff", type=int, default=80, help="Cutoff for ILP during Hybrid algorithm")
    parser.add_argument("--time_limit", type=int, default=1500, help="Time limit for ILP convergence")
    parser.add_argument("--greedy", "-g", action="store_true", default=False)
    parser.add_argument("--camin-sokal", "-cs", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False, help="output verbosity")
    parser.add_argument("--mutation_map", type=str, default="")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--max_neighborhood_size", type=int, default=10000)

    args = parser.parse_args()

    char_fp = args.char_fp
    out_fp = args.out_fp
    verbose = args.verbose

    cutoff = args.cutoff
    time_limit = args.time_limit
    num_threads = args.num_threads

    max_neighborhood_size = args.max_neighborhood_size

    stem = ''.join(char_fp.split(".")[:-1])

    cm = pd.read_csv(char_fp, sep='\t', index_col=0, dtype=str)
    cm_uniq = cm.drop_duplicates(inplace=False)

    cm_lookup = list(cm.apply(lambda x: "|".join(x.values), axis=1))
    newick = ""

    prior_probs = None
    if args.mutation_map != "":

        prior_probs = read_mutation_map(args.mutation_map)

    if args.greedy:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print('Running Greedy Algorithm on ' + str(len(target_nodes)) + " Cells")


        #string_to_sample = dict(zip(target_nodes, cm_uniq.index))

        #target_nodes = list(map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index))

        reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="greedy", prior_probabilities=prior_probs)
        
        net = reconstructed_network_greedy.get_network()

        root = [n for n in net if net.in_degree(n) == 0][0]
        # score parsimony
        score = 0
        for e in nx.dfs_edges(net, source=root):
            score += e[0].get_edit_distance(e[1])
           
        print("Parsimony: " + str(score))
        
        #reconstructed_network_greedy = nx.relabel_nodes(reconstructed_network_greedy, string_to_sample)
        #newick = convert_network_to_newick_format(reconstructed_network_greedy) 
        newick = reconstructed_network_greedy.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_greedy, open(out_stem + ".pkl", "wb")) 

    elif args.hybrid:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print('Running Hybrid Algorithm on ' + str(len(target_nodes)) + " Cells")
            print('Parameters: ILP on sets of ' + str(cutoff) + ' cells ' + str(time_limit) + 's to complete optimization') 

        #string_to_sample = dict(zip(target_nodes, cm_uniq.index))

        #target_nodes = list(map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index))

        print("running algorithm...")
        reconstructed_network_hybrid = solve_lineage_instance(target_nodes, method="hybrid", hybrid_subset_cutoff=cutoff, prior_probabilities=prior_probs, time_limit=time_limit, threads=num_threads, max_neighborhood_size=max_neighborhood_size)

        if verbose:
            print("Scoring Parsimony...")
            
        net = reconstructed_network_hybrid.get_network()
        # score parsimony
        #score = 0
        #for e in net.edges():
        #    score += get_edge_length(e[0], e[1])
           
        #print("Parsimony: " + str(score))
        
        newick = reconstructed_network_hybrid.get_newick()
        #if verbose:
        #    print("Parsimony: " + str(score))
        
        if verbose:
            print("Writing the tree to output...")

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_hybrid, open(out_stem + ".pkl", "wb")) 

        with open(out_fp, "w") as f:
            f.write(newick)


    elif args.ilp:

        target_nodes = list(cm_uniq.apply(lambda x: Node(x.name, x.values), axis=1))

        if verbose:
            print("Running ILP Algorithm on " + str(len(target_nodes)) + " Unique Cells")
            print("Paramters: ILP allowed " + str(time_limit) + "s to complete optimization")

        reconstructed_network_ilp = solve_lineage_instance(target_nodes, method="ilp", prior_probabilities=prior_probs, time_limit=time_limit, max_neighborhood_size=max_neighborhood_size)

        net = reconstructed_network_ilp.get_network()

        root = [n for n in net if net.in_degree(n) == 0][0]

        # score parsimony
        score = 0
        for e in nx.dfs_edges(net, source=root):
            score += e[0].get_edit_distance(e[1])
           
        print("Parsimony: " + str(score))
        
        newick = reconstructed_network_ilp.get_newick()
        if verbose:
            print("Parsimony: " + str(score))
        
        if verbose:
            print("Writing the tree to output...")

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_ilp, open(out_stem + ".pkl", "wb")) 

        with open(out_fp, "w") as f:
            f.write(newick)


    elif args.neighbor_joining:

        if verbose:
            print("Running Neighbor-Joining on " + str(cm_uniq.shape[0]) + " Unique Cells")

        fn = stem + "phylo.txt"
        infile = stem + "infile.txt"
        
        cm_uniq.to_csv(fn, sep='\t')
        
        script = (SCLT_PATH / 'TreeSolver' / 'binarize_multistate_charmat.py')
        cmd =  "python3.6 " + str(script) +  " " + fn + " " + infile + " --relaxed" 
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)
        
        aln = AlignIO.read(infile, "phylip-relaxed")

        calculator = DistanceCalculator('identity')
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

        nj_net = fill_in_tree(nj_net, cm)

        # nj_net = tree_collapse2(nj_net)

        rdict = {}
        for n in nj_net: 
            if nj_net.out_degree(n) == 0 and n.char_string in cm_lookup:
                n.is_target = True
            else:
                n.is_target = False

        state_tree = nj_net
        ret_tree =  Cassiopeia_Tree(method='neighbor-joining', network=state_tree, name='Cassiopeia_state_tree')

        out_stem = "".join(out_fp.split(".")[:-1])

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb")) 

        newick = dp.convert_network_to_newick_format(nj_net)

        with open(out_fp, "w") as f:
           f.write(newick)

        os.system("rm " + infile)
        os.system("rm " + fn)

    elif args.neighbor_joining_weighted:

        if verbose:
            print("Running Neighbor-Joining with Weighted Scoring on " + str(cm_uniq.shape[0]) + " Unique Cells")

        dm = compute_distance_mat(cm_uniq.values.astype(np.str), cm_uniq.shape[0], priors=prior_probs)
        
        ids = cm_uniq.index 
        dm = sp.spatial.distance.squareform(dm)

        dm = DistanceMatrix(dm, ids)

        newick_str = nj(dm, result_constructor=str)

        tree = dp.newick_to_network(newick_str)

        nj_net = fill_in_tree(tree, cm_uniq)
        #nj_net = tree_collapse2(nj_net)
        
        out_stem = "".join(out_fp.split(".")[:-1])

        rdict = {}
        for n in nj_net: 
            if nj_net.out_degree(n) == 0 and n.char_string in cm_lookup:
                n.is_target = True
            else:
                n.is_target = False

        state_tree = nj_net
        ret_tree =  Cassiopeia_Tree(method='neighbor-joining', network=state_tree, name='Cassiopeia_state_tree')

        #ret_tree.collapse_edges()

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb"))

        newick = ret_tree.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

    elif args.camin_sokal:
        
        cells = cm_uniq.index
        samples = [("s" + str(i)) for i in range(len(cells))]
        samples_to_cells = dict(zip(samples, cells))
        
        cm_uniq.index = list(range(len(cells)))
        
        if verbose:
            print("Running Camin-Sokal on " + str(cm_uniq.shape[0]) + " Unique Cells")


        infile = stem + 'infile.txt'
        fn = stem + "phylo.txt"
        weights_fn = stem + "weights.txt"
        
        cm_uniq.to_csv(fn, sep='\t')

        script = (SCLT_PATH / 'TreeSolver' / 'binarize_multistate_charmat.py')
        cmd =  "python3.6 " + str(script) +  " " + fn + " " + infile
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

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
        FH.write("Y\n")
        FH.write("F\n" + outtree + "\n")
        FH.close()

        t0 = time.time()
        cmd = "~/software/phylip-3.697/exe/mix"
        cmd += " < " + responses + " > screenout" 
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

        consense_outtree = stem + "consenseouttree.txt"
        consense_outfile = stem + "conenseoutfile.txt"

        FH = open(responses, "w")
        FH.write(outtree + "\n")
        FH.write("F\n" + consense_outfile + "\n")
        FH.write("Y\n")
        FH.write("F\n" + consense_outtree + "\n")
        FH.close()

        if verbose:
            print("Computing Consensus Tree, elasped time: " + str(time.time() - t0))

        cmd = "~/software/phylip-3.697/exe/consense"
        cmd += " < " + responses + " > screenout2" 
        p2 = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p2.pid, 0)

        newick_str = ""
        with open(consense_outtree, "r") as f:
            for l in f:
                l = l.strip()
                newick_str += l

        #tree = Phylo.parse(consense_outtree, "newick").next()
        tree = dp.newick_to_network(newick_str)

        #tree.rooted = True
        #cs_net = tree_collapse(tree)
        #cs_net = Phylo.to_networkx(tree)

        cs_net = nx.relabel_nodes(tree, samples_to_cells)

        cs_net = fill_in_tree(cs_net, cm)

        #cs_net = tree_collapse2(cs_net)

        out_stem = "".join(out_fp.split(".")[:-1])

        # rdict = {}
        for n in cs_net:
            if n.char_string in cm_lookup:
                n.is_target = True
            # spl = n.split("_")
            # nn = Node('state-node', spl[0].split("|"), is_target = False)
            # if len(spl) > 1:
            #     nn.pid = spl[1] 
            # if spl[0] in cm.index.values:
            #     nn.is_target = True
            # rdict[n] = nn

        # state_tree = nx.relabel_nodes(cs_net, rdict)
        state_tree = cs_net
        ret_tree =  Cassiopeia_Tree(method='camin-sokal', network=state_tree, name='Cassiopeia_state_tree')

        pic.dump(ret_tree, open(out_stem + ".pkl", "wb"))

        newick = dp.convert_network_to_newick_format(state_tree)
        # newick = ret_tree.get_newick()

        with open(out_fp, "w") as f:
            f.write(newick)

        os.system("rm " + outfile)
        os.system("rm " + responses)
        os.system("rm " + outtree)
        os.system("rm " + consense_outfile)
        os.system("rm " + infile)
        os.system("rm " + fn)
    
    elif alg == "--max-likelihood" or alg == '-ml':

        #cells = cm.index
        #samples = [("s" + str(i)) for i in range(len(cells))]
        #samples_to_cells = dict(zip(samples, cells))
        
        #cm.index = list(range(len(cells)))
        
        if verbose:
            print("Running Maximum Likelihood on " + str(cm.shape[0]) + " Unique Cells")

        infile = stem + 'infile.txt'
        fn = stem + "phylo.txt"
        
        cm.to_csv(fn, sep='\t')

        script = (SCLT_PATH / 'TreeSolver' / 'binarize_multistate_charmat.py')
        cmd =  "python3.6 " + str(script) +  " " + fn + " " + infile + " --relaxed" 
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
        
        raise Exception("Please choose an algorithm from the list: greedy, hybrid, ilp, nj, max-likelihood, or camin-sokal")

if __name__ == "__main__":
    main()
