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

import Bio.Phylo as Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, ParsimonyScorer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
import networkx as nx

import sys
import os

sys.path.append("/home/mattjones/projects/scLineages/scripts")
sys.path.append("/home/mattjones/projects/scLineages/SingleCellLineageTracing/Alex_Solver")

from data_pipeline import convert_network_to_newick_format, newick_to_network
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
    param = spl[-3]
    run = spl[-1].split(".")[0]

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

    unique_ii = np.unique(target_nodes, return_index=True)
    target_nodes_uniq = np.array(target_nodes)[unique_ii[1]]
    target_nodes_original_network_uniq = np.array(target_nodes_original_network)[unique_ii[1]]


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
            print("Running Neighbor-Joining on " + str(len(target_nodes_uniq)) + " Unique Cells")


        infile = ''.join(name.split(".")[:-1]) + 'infile.txt'
        fn = ''.join(name.split(".")[:-1]) + 'phylo.txt'
        write_leaves_to_charmat(target_nodes_original_network_uniq, fn)

        os.system("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py " + fn + " " + infile)
        aln = AlignIO.read(infile, "phylip")

        aln = unique_alignments(aln)

        calculator = DistanceCalculator('identity')
        constructor = DistanceTreeConstructor(calculator, 'nj')

        t0 = time.time()
        tree = constructor.build_tree(aln)
        reconstructed_network = Phylo.to_networkx(tree)

        # convert labels to strings, not Bio.Phylo.Clade objects
        c2str = map(lambda x: str(x), cs_net.nodes())
        c2strdict = dict(zip(cs_net.nodes(), c2str))
        cs_net = nx.relabel_nodes(cs_net, c2strdict)

        # convert labels to characters for triplets correct analysis
        cs_net = nx.relabel_nodes(cs_net, s_to_char)

        tp = check_triplets_correct(true_network, cs_net)

        print(str(param) + "\t" + str(run) + "\t" + str(tp) + "\t" + "neighbor-joining" + "\t" + t + '\t' + str(time.time() - t0))

    elif alg == "--camin-sokal" or alg == "-cs":
        
        if verbose:
            print('Running Camin-Sokal Max Parsimony Algorithm on ' + str(len(target_nodes_uniq)) + " Unique Cells")

        infile = ''.join(name.split(".")[:-1]) + 'infile.txt'
        fn = ''.join(name.split(".")[:-1]) + 'phylo.txt'
        weights_fn = ''.join(name.split(".")[:-1]) + "weights.txt"
        write_leaves_to_charmat(target_nodes_original_network_uniq, fn)
        
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
        FH.write("W\n")
        FH.write("Y\n")
        FH.write(weights_fn + "\n")
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
        cmd += " < " + responses + " > screenout" 
        p2 = subprocess.Popen(cmd, shell=True)
        pid, ecode = os.waitpid(p2.pid, 0)

        # read in newick file to networkx format
        cs_net = newick_to_network(consense_outtree, f=0) 

        # convert labels to characters for triplets correct analysis
        cs_net = nx.relabel_nodes(cs_net, s_to_char)

        tp = check_triplets_correct(true_network, cs_net)

        print(str(param) + "\t" + str(run) + "\t" + str(tp) + "\t" + "camin-sokal" + "\t" + t + "\t" + str(time.time() - t0))

        os.system("rm " + outfile)
        os.system("rm " + responses)
        os.system("rm " + outtree)
        os.system("rm " + consense_outfile)
        os.system("rm " + infile)
        os.system("rm " + fn)

    else:
        
        raise Exception("Please choose an algorithm from the list: greedy, hybrid, ilp, nj, or camin-sokal")


