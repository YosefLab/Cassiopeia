from __future__ import division

import subprocess
from string import ascii_uppercase

import numpy as np
import pandas as pd
import pandascharm as pc
import random
from pylab import *
import pickle as pic

import argparse

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer, DistanceTreeConstructor
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import networkx as nx

import sys
import os

from SingleCellLineageTracing.TreeSolver.lineage_solver import *
from SingleCellLineageTracing.TreeSolver.simulation_tools import *
from SingleCellLineageTracing.TreeSolver import *

def write_leaves_to_charmat(target_nodes, fn):
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
    
    labund = np.array(map(lambda x: float(-1 * np.log2(x)) if x > 1 else x, abund))
    labund[labund == 0] = labund.min()

    # scale linearly to range for phylip weights
    _min = 0
    _max = 35

    scaled = (_max - _min) / (labund.max() - labund.min()) * (labund - labund.max()) + _max
    scaled = map(lambda x: int(x), scaled)

    weights_range = [str(i) for i in range(10)] + [l for l in ascii_uppercase]
    weights_dict = dict(zip(range(36), weights_range))

    scaled = map(lambda x: weights_dict[x], scaled)

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

    cm = pd.read_csv(char_fp, sep='\t', index_col=0)
    cm_uniq = cm.drop_duplicates(inplace=False)

    newick = ""

    prior_probs = None
    if args.mutation_map != "":

        prior_probs = read_mutation_map(args.mutation_map)

    if args.greedy:

        target_nodes = cm_uniq.astype(str).apply(lambda x: '|'.join(x), axis=1)

        if verbose:
            print('Running Greedy Algorithm on ' + str(len(target_nodes)) + " Cells")


        string_to_sample = dict(zip(target_nodes, cm.index))

        target_nodes = map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index)

        reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="greedy", prior_probabilities=prior_probs)
        
        # score parsimony
        score = 0
        for e in reconstructed_network_greedy.edges():
            score += get_edge_length(e[0], e[1])
           
        print("Parsimony: " + str(score))
        
        #reconstructed_network_greedy = nx.relabel_nodes(reconstructed_network_greedy, string_to_sample)
        newick = convert_network_to_newick_format(reconstructed_network_greedy) 

        with open(out_fp, "w") as f:
            f.write(newick)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_greedy, open(out_stem + ".pkl", "wb")) 

    elif args.hybrid:

        target_nodes = cm_uniq.astype(str).apply(lambda x: '|'.join(x), axis=1)

        if verbose:
            print('Running Hybrid Algorithm on ' + str(len(target_nodes)) + " Cells")
            print('Parameters: ILP on sets of ' + str(cutoff) + ' cells ' + str(time_limit) + 's to complete optimization') 

        string_to_sample = dict(zip(target_nodes, cm.index))

        target_nodes = map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index)

        print("running algorithm...")
        reconstructed_network_hybrid = solve_lineage_instance(target_nodes, method="hybrid", hybrid_subset_cutoff=cutoff, prior_probabilities=prior_probs, time_limit=time_limit, threads=num_threads, max_neighborhood_size=max_neighborhood_size)

        if verbose:
            print("Scoring Parsimony...")
            
        # score parsimony
        score = 0
        for e in reconstructed_network_hybrid.edges():
            score += get_edge_length(e[0], e[1])
           
        if verbose:
            print("Parsimony: " + str(score))
        
        if verbose:
            print("Writing the tree to output...")

        #reconstructed_network_hybrid = nx.relabel_nodes(reconstructed_network_hybrid, string_to_sample)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_hybrid, open(out_stem + ".pkl", "wb")) 

        newick = convert_network_to_newick_format(reconstructed_network_hybrid) 

        with open(out_fp, "w") as f:
            f.write(newick)


    elif args.ilp:

        target_nodes = cm_uniq.astype(str).apply(lambda x: '|'.join(x), axis=1)

        if verbose:
            print("Running ILP Algorithm on " + str(len(target_nodes)) + " Unique Cells")
            print("Paramters: ILP allowed " + str(time_limit) + "s to complete optimization")

        string_to_sample = dict(zip(target_nodes, cm.index))

        target_nodes = map(lambda x, n: x + "_" + n, target_nodes, cm_uniq.index)

        reconstructed_network_ilp = solve_lineage_instance(target_nodes, method="ilp", prior_probabilities=prior_probs, time_limit=time_limit, max_neighborhood_size=max_neighborhood_size)

        # score parsimony
        score = 0
        for e in reconstructed_network_ilp.edges():
            score += get_edge_length(e[0], e[1])
           
        print("Parsimony: " + str(score))
        
        #reconstructed_network_ilp = nx.relabel_nodes(reconstructed_network_ilp, string_to_sample)
        newick = convert_network_to_newick_format(reconstructed_network_ilp) 

        with open(out_fp, "w") as f:
            f.write(newick)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_ilp, open(out_stem + ".pkl", "wb")) 

    elif args.neighbor_joining:


        cm.drop_duplicates(inplace=True) 

        if verbose:
            print("Running Neighbor-Joining on " + str(cm.shape[0]) + " Unique Cells")

        fn = stem + "phylo.txt"
        infile = stem + "infile.txt"
        
        cm.to_csv(fn, sep='\t')
        
        os.system("python2 ~/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py "  + fn + " " + infile + " --relaxed")
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
        c2str = map(lambda x: x.name, nj_net.nodes())
        c2strdict = dict(zip(nj_net.nodes(), c2str))
        nj_net = nx.relabel_nodes(nj_net, c2strdict)

        nj_net = tree_collapse(nj_net)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(nj_net, open(out_stem + ".pkl", "wb")) 

        newick = convert_network_to_newick_format(nj_net)

        with open(out_fp, "w") as f:
            f.write(newick)

        os.system("rm " + infile)
        os.system("rm " + fn)

    elif args.camin_sokal:
        
        cells = cm.index
        samples = [("s" + str(i)) for i in range(len(cells))]
        samples_to_cells = dict(zip(samples, cells))
        
        cm.index = list(range(len(cells)))
        
        if verbose:
            print("Running Camin-Sokal on " + str(cm.shape[0]) + " Unique Cells")


        infile = stem + 'infile.txt'
        fn = stem + "phylo.txt"
        weights_fn = stem + "weights.txt"
        
        cm.to_csv(fn, sep='\t')

        os.system("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py " + fn + " " + infile) 

        weights = construct_weights(infile, weights_fn)

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
        tree = newick_to_network(newick_str)
        #tree.rooted = True
        cs_net = tree_collapse(tree)
        #cs_net = Phylo.to_networkx(tree)

        cs_net = nx.relabel_nodes(cs_net, samples_to_cells)

        out_stem = "".join(out_fp.split(".")[:-1])

        pic.dump(cs_net, open(out_stem + ".pkl", "wb"))

        newick = convert_network_to_newick_format(cs_net)

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
            print("Running Camin-Sokal on " + str(cm.shape[0]) + " Unique Cells")

        infile = stem + 'infile.txt'
        fn = stem + "phylo.txt"
        
        cm.to_csv(fn, sep='\t')

        os.system("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py " + fn + " " + infile + " --relaxed") 

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
