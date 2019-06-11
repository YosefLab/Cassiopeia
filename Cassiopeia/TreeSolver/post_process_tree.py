from __future__ import division
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
sys.setrecursionlimit(10000)
import pickle as pic

import argparse

import networkx as nx
from collections import defaultdict
import pylab

import copy

from .data_pipeline import *
from Cassiopeia.TreeSolver.Node import Node

def prune_and_clean_leaves(G):
    """
    Prune off leaves that don't correspond to samples and clean up the names on leaves (i.e. only keep the sample
    labels and remove character states or post-processing hashes.)

    :param G: 
        Networkx Graph as a tree
    :return: 
        Pruned and cleaned tree as a Networkx object.
    """

    new_nodes = []
    new_edges = []


    def prune_leaves(G):

        nodes_to_remove = []

        root = [n for n in G if G.in_degree(n) == 0][0]

        # first remove paths to leaves that don't correspond to samples
        _leaves = [n for n in G if G.out_degree(n) == 0]

        for n in _leaves:
            # if we have this case, where the leaf doesn't have a sample label in the name, we need to remove this path
            #if "target" not in n:
            if not n.is_target:
                nodes_to_remove.append(n)

        return nodes_to_remove

    nodes_to_remove = prune_leaves(G)
    while len(nodes_to_remove) > 0:
        for n in set(nodes_to_remove):
            G.remove_node(n)

        nodes_to_remove = prune_leaves(G)

    # remove character strings from node name
    # node_dict = {}
    # for n in tqdm(G.nodes, desc="removing character strings from sample names"):
    #     spl = n.split("_")
    #     if "|" in spl[0] and "target" in n:
    #         nn = "_".join(spl[1:])
    #         node_dict[n] = nn

    # G = nx.relabel_nodes(G, node_dict)

    node_dict2 = {}
    for n in G.nodes:
       # spl = n.split("_")
        if n.is_target:
            #if spl[-1] == "target":
            #    name = "_".join(spl[:-1])
            #else:
            #    name = "_".join(spl[:-2])

            # if this target is a leaf, just rename it
            # else we must add an extra 'redundant' leaf here
            if G.out_degree(n) != 0:
            #    node_dict2[n] = name
                n.is_target = False
                new_node = Node(n.name, n.get_character_vec(), is_target=True)
            # else:
                new_nodes.append(new_node)
                new_edges.append((n, new_node))

    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    # G = nx.relabel_nodes(G, node_dict2)

    return G

def assign_samples_to_charstrings(G, cm):
    """
    Preprocessing step if sample names are not in the tree. Assigns sample name to appropriate
    character states in the phylogeny. 

    :param G:
        Input graph.
    :param cm:
        Character matrix pandas Dataframe.

    :return:
        Networkx Graph object as a tree with samples mapped onto the tree.
    """

    new_nodes = []
    new_edges = []

    nodes_to_remove = []

    root = [n for n in G if G.in_degree(n) == 0][0]

    cm["lookup"] = cm.astype(str).apply(lambda x: "|".join(x), axis=1)

    for n in G:

        if n.get_character_string() in cm['lookup'].values and n.is_target:
            n.is_target = False
            sub_cm  = cm.loc[cm["lookup"] == n.get_character_string()]
            _nodes = sub_cm.apply(lambda x: Node(x.name, x.values, is_target=True), axis=1)
            for new_node in _nodes:
                new_nodes.append(new_node)
                new_edges.append((n, new_node))

    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    return G

def tree_collapse(graph):
    """
    Given a networkx graph in the form of a tree, collapse two nodes togethor if there are no mutations seperating the two nodes
        :param graph: Networkx Graph as a tree
        :return: Collapsed tree as a Networkx object
    """

    new_network = nx.DiGraph()
    for edge in graph.edges():
        if edge[0].split('_')[0] == edge[1].split('_')[0]:
            if graph.out_degree(edge[1]) != 0:
                for node in graph.successors(edge[1]):
                    new_network.add_edge(edge[0], node)
            else:
                new_network.add_edge(edge[0], edge[1])
        else:
            new_network.add_edge(edge[0], edge[1])
    return new_network

def add_redundant_leaves(G, cm):
    """
    To fairly take into account sample purity, we'll add back in 'redundant' leaves (i.e.
    leaves that were removed because of non-unique character strings).

    :param G:
        Input graph
    :param cm:
        Character matrix pandas Dataframe

    :return:
        Graph with redundant samples added back on.
    """

    # create lookup value for duplicates
    cm["lookup"] = cm.astype('str').apply('|'.join, axis=1)
    net_nodes = np.intersect1d(cm.index, [n.name for n in G])

    uniq = cm.loc[net_nodes]

    if uniq.shape == cm.shape:
        return G

    # find all non-unique character states in cm
    #nonuniq = np.setdiff1d(cm.index, np.array(uniq))
    nonuniq = np.setdiff1d(cm.index, uniq.index)

    for n in nonuniq:

        new_node = str(n)

        try:
            _leaf = uniq.index[uniq["lookup"] == cm.loc[n]["lookup"]][0]

            new_node = Node(str(n), cm.loc[n].values, is_target=True)

            parents = list(G.predecessors(_leaf))
            for p in parents:
                G.add_edge(p, new_node)
        except:
            continue


    return G

def post_process_tree(G, cm, alg):
    """
    Entry point for post-process-tree. Depending on which algorithm was used to construct a tree, 
    will perform sample mapping (i.e. assigning samples to the character states in the phylogeny)

    :param G:
        Input graph.
    :param cm:
        Character matrix pandas Dataframe
    :param alg:
        Which algorithm was used. Chosen from `greedy, hybrid, ilp, neighbor-joining`, or `camin-sokal`.

    :return:
        Post-Processed Tree as a networkx object
    """

    
    if alg in ['greedy', 'hybrid', 'ilp']:
        G = assign_samples_to_charstrings(G, cm)
        G = prune_and_clean_leaves(G)

    G = add_redundant_leaves(G, cm)

    return G

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("treefp", type=str, help="Cassiopeia Tree Pickle File")
    parser.add_argument("char_fp", type=str, help="Character matrix")
    parser.add_argument("out_fp", type=str, help="Output file -- will be written as a newick file!")

    args = parser.parse_args()
    treefp = args.treefp
    char_fp = args.char_fp
    out_fp = args.out_fp

    if out_fp.split(".")[-1] != 'txt':

        print("Warning! output is a newick file")

    tree = pic.load(open(treefp, "rb"))
    cm = pd.read_csv(char_fp, sep='\t', index_col = 0)

    tree2 = tree.post_process(cm = cm)
    G = tree2.network

    stem = ".".join(out_fp.split(".")[:-1])

    pic.dump(tree2, open(stem + ".pkl", "wb"))

    newick = convert_network_to_newick_format(G)

    with open(out_fp, "w") as f:
        f.write(newick)

if __name__ == "__main__":
    main()
