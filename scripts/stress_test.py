from __future__ import division
from __future__ import print_function

import subprocess
from time import sleep

import numpy as np
import pandas as pd
import random
from pylab import *
import pickle as pic

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
import networkx as nx

import sys
import os

sys.path.append("/home/mattjones/projects/scLineages/scripts")
sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from data_pipeline import convert_network_to_newick_format
from lineage_solver.lineage_solver import solve_lineage_instance
from lineage_solver.solution_evaluation_metrics import cci_score
from simulation_tools.dataset_generation import generate_simulated_full_tree
from simulation_tools.simulation_utils import get_leaves_of_tree
from simulation_tools.validation import check_triplets_correct

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

if __name__ == "__main__":
    """
    Takes in a character matrix, an algorithm, and an output file and 
    returns a tree in newick format.

    """

    netfp = sys.argv[1]
    alg = sys.argv[2]
    t = sys.argv[3]
    verbose_command = sys.argv[4] if len(sys.argv) > 4 else ""
    
    name = netfp.split("/")[-1]
    spl = name.split("_")
    param = spl[3]
    run = spl[5].split(".")[0]

    stem = ''.join(name.split(".")[:-1])

    verbose = False
    if verbose_command == "--verbose":
        verbose=True

    true_network = pic.load(open(netfp, "rb"))
    
    target_nodes = get_leaves_of_tree(true_network, clip_identifier=True)
    target_nodes_original_network = get_leaves_of_tree(true_network, clip_identifier=False)

    k = map(lambda x: "s" + x.split("_")[-1], target_nodes_original_network)
    s_to_char = dict(zip(k, target_nodes))
    char_to_s = dict(zip(target_nodes, k))

    if alg == "--greedy" or alg == "-g":

        if verbose:
            print('Running Greedy Algorithm on ' + str(len(target_nodes)) + " Cells")

        write_leaves_to_charmat(target_nodes_original_network)

        os.system("python2 binarize_multistate_charmat.py phylo.txt infile")

        reconstructed_network_greedy = solve_lineage_instance(target_nodes, method="greedy")

        tp = check_triplets_correct(true_network, reconstructed_network_greedy)
        print(str(param) + "\t" + str(run) + "\t" + str(tp) + "\t" + "greedy" + "\t" + t)

        if verbose:
            reconstructed_network_greedy = nx.relabel_nodes(reconstructed_network_greedy, char_to_s)
            newick = convert_network_to_newick_format(reconstructed_network_greedy) 
            with open("test_newick", "w") as f:
                f.write(newick)

    elif alg == "--hybrid":

        print('hybrid')
        with open(out_fp, "w") as f:
            f.write(newick + ";")

    elif alg == '--ilp':

        print('ilp')
        with open(out_fp, "w") as f:
            f.write(newick + ";")

    elif alg == '--neighbor-joining' or alg == "-nj":
        
        if verbose:
            print("Running Neighbor-Joining on " + str(len(target_nodes)) + " Cells")

        write_leaves_to_charmat(target_nodes_original_network, fn)

        infile = ''.join(name.split(".")[:-1]) + 'infile.txt'
        fn = ''.join(name.split(".")[:-1]) + 'phylo.txt'

        os.system("python2 binarize_multistate_charmat.py " + fn + " " + infile)
        aln = AlignIO.read(infile, "phylip")
        calculator = DistanceCalculator('identity')
        constructor = DistanceTreeConstructor(calculator, 'nj')
        tree = constructor.build_tree(aln)
        cs_net = Phylo.to_networkx(tree)

        # convert labels to strings, not Bio.Phylo.Clade objects
        c2str = map(lambda x: str(x), cs_net.nodes())
        c2strdict = dict(zip(cs_net.nodes(), c2str))
        cs_net = nx.relabel_nodes(cs_net, c2strdict)

        # convert labels to characters for triplets correct analysis
        cs_net = nx.relabel_nodes(cs_net, s_to_char)

        tp = check_triplets_correct(true_network, cs_net)

        print(str(param) + "\t" + str(run) + "\t" + str(tp) + "\t" + "neighbor-joining" + "\t" + t)

        if verbose:
            print("Parsimony score: ", end="", flush=True)
            scorer = ParsimonyScorer()
            print(scorer.get_score(tree, aln))

    elif alg == "--camin-sokal" or alg == "-cs":
        
        if verbose:
            print('Running Camin-Sokal Max Parsimony Algorithm on ' + str(len(target_nodes)) + " Cells")

        infile = ''.join(name.split(".")[:-1]) + 'infile.txt'
        fn = ''.join(name.split(".")[:-1]) + 'phylo.txt'
        write_leaves_to_charmat(target_nodes_original_network, fn)
        
        os.system("python2 /home/mattjones/projects/scLineages/scripts/binarize_multistate_charmat.py " + fn + " " + infile) 

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

        # convert labels to strings, not Bio.Phylo.Clade objects
        c2str = map(lambda x: str(x), cs_net.nodes())
        c2strdict = dict(zip(cs_net.nodes(), c2str))
        cs_net = nx.relabel_nodes(cs_net, c2strdict)

        # convert labels to characters for triplets correct analysis
        cs_net = nx.relabel_nodes(cs_net, s_to_char)

        tp = check_triplets_correct(true_network, cs_net)

        print(str(param) + "\t" + str(run) + "\t" + str(tp) + "\t" + "camin-sokal" + "\t" + t)

    else:
        
        raise Exception("Please choose an algorithm from the list: greedy, hybrid, ilp, nj, or camin-sokal")


