from __future__ import division

import subprocess

import numpy as np
import pandas as pd
import random
from pylab import *
import pickle as pic

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer, DistanceTreeConstructor
from Bio import AlignIO
import networkx as nx

import sys
import os

sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from data_pipeline import convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
from simulation_tools.simulation_utils import get_leaves_of_tree

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

if __name__ == "__main__":
    """
    Takes in a charachter matrix, an algorithm, and an output file and 
    returns a tree in newick format.

    """

    char_fp = sys.argv[1]
    alg = sys.argv[2]
    out_fp = sys.argv[3]
    verbosity = "" if len(sys.argv) < 4 else sys.argv[4]

    stem = ''.join(char_fp.split(".")[:-1])

    verbose = False
    if verbosity == "--verbose":
        verbose = True

    cm = pd.read_csv(char_fp, sep='\t', index_col=0)

    newick = ""

    if alg == "--greedy" or alg == "-g":

        target_nodes = cm.apply(lambda x: '|'.join(x), axis=1)

        if verbose:
            print('Running Greedy Algorithm on ' + str(len(target_nodes)) + " Cells")


        string_to_sample = dict(zip(target_nodes, cm.index))

        reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="greedy")
        
        reconstructed_network_greedy = nx.relabel_nodes(reconstructed_network_greedy, string_to_sample)
        newick = convert_network_to_newick_format(reconstructed_network_greedy) 

        with open(out_fp, "w") as f:
            f.write(newick)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_greedy, open(out_stem + ".pkl", "wb")) 

    elif alg == "--hybrid" or alg == "-h":

        target_nodes = cm.apply(lambda x: '|'.join(x), axis=1)

        if verbose:
            print('Running Hybrid Algorithm on ' + str(len(target_nodes)) + " Cells")
            print('Default Parameters: ILP on sets of 50 cells, 900s to complete optimization') 

        string_to_sample = dict(zip(target_nodes, cm.index))

        reconstructed_network_hybrid = solve_lineage_instance(target_nodes, method="hybrid", hybrid_subset_cutoff=50)
        
        reconstructed_network_hybrid = nx.relabel_nodes(reconstructed_network_hybrid, string_to_sample)
        newick = convert_network_to_newick_format(reconstructed_network_hybrid) 

        with open(out_fp, "w") as f:
            f.write(newick)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(reconstructed_network_hybrid, open(out_stem + ".pkl", "wb")) 

    elif alg == '--ilp':

        print('ilp')
        with open(out_fp, "w") as f:
            f.write(newick)

    elif alg == '--neighbor-joining' or alg == '-nj':


        cells = cm.index
        samples = [("s" + str(i)) for i in range(len(cells))]
        samples_to_cells = dict(zip(samples, cells))
        cm.index = list(range(len(cells)))

        if verbose:
            print("Running Neighbor-Joining on " + str(len(cells)) + " Cells")
        
        cm.to_csv("phylo.txt", sep='\t')
        

        os.system("python2 ~/projects/scLineages/scripts/binarize_multistate_charmat.py phylo.txt infile")
        aln = AlignIO.read("infile", "phylip")

        calculator = DistanceCalculator('identity')
        constructor = DistanceTreeConstructor(calculator, 'nj')
        tree = constructor.build_tree(aln)
        tree.rooted = True # force rootedness just in case
        nj_net = Phylo.to_networkx(tree)

        # convert labels to strings, not Bio.Phylo.Clade objects
        c2str = map(lambda x: str(x), nj_net.nodes())
        c2strdict = dict(zip(nj_net.nodes(), c2str))
        nj_net = nx.relabel_nodes(nj_net, c2strdict)

        # convert labels to characters for writing to file 
        nj_net = nx.relabel_nodes(nj_net, samples_to_cells)

        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(nj_net, open(out_stem + ".pkl", "wb")) 

        newick = convert_network_to_newick_format(nj_net) 
        

        with open(out_fp, "w") as f:
            f.write(newick)

    elif alg == "--camin-sokal" or alg == "-cs":
        
        cells = cm.index
        samples = [("s" + str(i)) for i in range(len(cells))]
        samples_to_cells = dict(zip(samples, cells))
        
        if verbose:
            print("Running Camin-Sokal on " + str(len(cells)) + " Cells")

        cm.index = list(range(len(cells)))
        
        cm.to_csv("phylo.txt", sep='\t')

        infile = stem + 'infile.txt'
        
        os.system("python2 /home/mattjones/projects/scLineages/scripts/binarize_multistate_charmat.py phylo.txt " + infile) 

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

        cmd = "~/software/phylip-3.697/exe/mix"
        cmd += " < " + responses + " > screenout" 
        p = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p.pid, 0)

        os.system("rm " + responses)

        tree = Phylo.parse(outtree, "newick").next()

        cs_net = Phylo.to_networkx(tree)
        c2str = map(lambda x: str(x), cs_net.nodes())
        c2strdict = dict(zip(cs_net.nodes(), c2str))
        cs_net = nx.relabel_nodes(cs_net, c2strdict)

        cs_net = nx.relabel_nodes(cs_net, samples_to_cells)
        out_stem = "".join(out_fp.split(".")[:-1])
        pic.dump(cs_net, open(out_stem + ".pkl", "wb"))

        newick = convert_network_to_newick_format(cs_net)

        with open(out_fp, "w") as f:
            f.write(newick)
    
    else:
        
        raise Exception("Please choose an algorithm from the list: greedy, hybrid, ilp, nj, or camin-sokal")


