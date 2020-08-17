from __future__ import division

import numpy as np
import pandas as pd
import pandascharm as pc
import random
from pylab import *
import pickle as pic
from pathlib import Path
import subprocess
from string import ascii_uppercase


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

from cassiopeia.TreeSolver.utilities import fill_in_tree, tree_collapse, newick_to_network
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

import cassiopeia as sclt

SCLT_PATH = Path(sclt.__path__[0])


def run_nj_naive(cm_uniq, stem, verbose = True):

    if verbose:
        print("Running Neighbor-Joining on " + str(cm_uniq.shape[0]) + " Unique Cells") 


    cm_lookup = list(cm_uniq.apply(lambda x: "|".join(x.values), axis=1))

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
    rndict = {}
    for n in nj_net:

        if n.name is None:
            rndict[n] = Node('state-node', [])
        elif n.name in cm_uniq:
            rndict[n] = Node(n.name, cm_uniq.loc[n.name].values)

    
    # convert labels to strings, not Bio.Phylo.Clade objects
    #c2str = map(lambda x: x.name, list(nj_net.nodes()))
    #c2strdict = dict(zip(list(nj_net.nodes()), c2str))
    nj_net = nx.relabel_nodes(nj_net, rndict)

    # nj_net = fill_in_tree(nj_net, cm_uniq)

    # nj_net = tree_collapse2(nj_net)

    rdict = {}
    for n in nj_net: 
        if nj_net.out_degree(n) == 0 and n.char_string in cm_lookup:
            n.is_target = True
        else:
            n.is_target = False

    state_tree = nj_net
    ret_tree =  Cassiopeia_Tree(method='neighbor-joining', network=state_tree, name='Cassiopeia_state_tree')

    os.system("rm " + infile)
    os.system("rm " + fn)

    return ret_tree

def run_nj_weighted(cm_uniq, prior_probs = None, verbose = True):

    if verbose:
        print("Running Neighbor-Joining with Weighted Scoring on " + str(cm_uniq.shape[0]) + " Unique Cells")

    cm_lookup = list(cm_uniq.apply(lambda x: "|".join(x.values), axis=1))

    dm = compute_distance_mat(cm_uniq.values.astype(np.str), cm_uniq.shape[0], priors=prior_probs)
    
    ids = cm_uniq.index 
    dm = sp.spatial.distance.squareform(dm)

    dm = DistanceMatrix(dm, ids)

    newick_str = nj(dm, result_constructor=str)

    tree = newick_to_network(newick_str, cm_uniq)

    nj_net = fill_in_tree(tree, cm_uniq)

    for n in nj_net: 
        if nj_net.out_degree(n) == 0 and n.char_string in cm_lookup:
            n.is_target = True
            n.name = 'state-node'
        else:
            n.is_target = False

    state_tree = nj_net
    ret_tree =  Cassiopeia_Tree(method='neighbor-joining', network=state_tree, name='Cassiopeia_state_tree')

    return ret_tree

def run_camin_sokal(cm_uniq, stem, verbose = True):

    stem = stem.split("/")[-1]

    cells = cm_uniq.index
    samples = [("s" + str(i)) for i in range(len(cells))]
    samples_to_cells = dict(zip(samples, cells))

    cm_lookup = list(cm_uniq.apply(lambda x: "|".join(x.values), axis=1))
    
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

    tree = dp.newick_to_network(newick_str, cm_uniq)

    cs_net = nx.relabel_nodes(tree, samples_to_cells)

    #cs_net = fill_in_tree(cs_net, cm_uniq)
    
    # rdict = {}
    for n in cs_net:
        if n.char_string in cm_lookup:
            n.is_target = True

    state_tree = cs_net
    ret_tree =  Cassiopeia_Tree(method='camin-sokal', network=state_tree, name='Cassiopeia_state_tree')

    os.system("rm " + outfile)
    os.system("rm " + responses)
    os.system("rm " + outtree)
    os.system("rm " + consense_outfile)
    os.system("rm " + infile)
    os.system("rm " + fn)

    return ret_tree


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
    
    return (d*len(s1)/num_present)

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

