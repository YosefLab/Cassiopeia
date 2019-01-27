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
from pylab import *

from SingleCellLineageTracing.TreeSolver import convert_network_to_newick_format
import SingleCellLineageTracing.TreeSolver.lineage_solver as ls

def prune_and_clean_leaves(G):
    """
    Given a networkx graph in the form of a tree, assign sample identities to character states.

    :param graph: Networkx Graph as a tree
    :return: postprocessed tree as a Networkx object
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
            if "target" not in n:
                nodes_to_remove.append(n)

        return nodes_to_remove

    nodes_to_remove = prune_leaves(G)
    while len(nodes_to_remove) > 0:
        for n in set(nodes_to_remove):
            G.remove_node(n)

        nodes_to_remove = prune_leaves(G)

    # remove character strings from node name
    node_dict = {}
    for n in tqdm(G.nodes, desc="removing character strings from sample names"):
        spl = n.split("_")
        if "|" in spl[0] and "target" in n:
            nn = "_".join(spl[1:])
            node_dict[n] = nn

    G = nx.relabel_nodes(G, node_dict)

    node_dict2 = {}
    for n in G.nodes:
        spl = n.split("_")
        if "target" in n:
            if spl[-1] == "target":
                name = "_".join(spl[:-1])
            else:
                name = "_".join(spl[:-2])

            # if this target is a leaf, just rename it
            # else we must add an extra 'redundant' leaf here
            if G.out_degree(n) == 0:
                node_dict2[n] = name
            else:
                new_nodes.append(name)
                new_edges.append((n, name))

    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    G = nx.relabel_nodes(G, node_dict2)

    return G

def assign_samples_to_charstrings(G, cm):

    new_nodes = []
    new_edges = []

    nodes_to_remove = []

    root = [n for n in G if G.in_degree(n) == 0][0]

    cm["lookup"] = cm.apply(lambda x: "|".join(x), axis=1)

    for n in G:

        if n in cm['lookup'].values:
            _nodes  = cm.loc[cm["lookup"] == n].index
            _nodes = map(lambda x: x + "_target", _nodes)
            for new_node in _nodes:
                new_nodes.append(new_node)
                new_edges.append((n, new_node))

    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    #_leaves = [n for n in G if G.out_degree(n) == 0]

    #for n in _leaves:
    #    if n not in cm.index:
    #        paths = nx.all_simple_paths(G, root, n)
    #        for p in paths:
    #            for n in p:
    #                if n != root and G.out_degree(n) <= 1:
    #                    nodes_to_remove.append(n)

    #for n in set(nodes_to_remove):
    #    G.remove_node(n)



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
    """

    # create lookup value for duplicates
    cm["lookup"] = cm.astype('str').apply('|'.join, axis=1)
    net_nodes = np.intersect1d(cm.index, [n for n in G])

    uniq = cm.loc[net_nodes]

    # find all non-unique character states in cm
    nonuniq = np.setdiff1d(cm.index, np.array(uniq))

    for n in nonuniq:

        new_node = str(n)

        try:
            _leaf = uniq.index[uniq["lookup"] == cm.loc[n]["lookup"]][0]

            parents = list(G.predecessors(_leaf))
            for p in parents:
                G.add_edge(p, new_node)
        except:
            continue


    return G

def post_process_tree(G, cm, alg):

    if alg == "greedy":
        G = assign_samples_to_charstrings(G, cm)
        G = prune_and_clean_leaves(G)

    if alg == "hybrid" or alg == "ilp":
        ls.clean_ilp_network(G)
        G = prune_and_clean_leaves(G)

    G = add_redundant_leaves(G, cm)

    return G

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("netfp", type=str, help="Networkx pickle file")
    parser.add_argument("char_fp", type=str, help="Character matrix")
    parser.add_argument("out_fp", type=str, help="Output file -- will be written as a newick file!")
    parser.add_argument("alg", type=str, help="Algorithm used to reconstruct this tree")

    algorithms = ["greedy", "hybrid", "ilp", "neighbor-joining", "camin-sokal"]

    args = parser.parse_args()
    netfp = args.netfp
    char_fp = args.char_fp
    alg = args.alg
    out_fp = args.out_fp

    if alg not in algorithms:
        raise Exception("Algorithm not recognized, use one of: " + str(algorithms))

    if out_fp.split(".")[-1] != 'txt':

        print("Warning! output is a newick file")

    G = nx.read_gpickle(netfp)
    cm = pd.read_csv(char_fp, sep='\t', index_col = 0)

    G = post_process_tree(G, cm, alg)

    stem = ".".join(out_fp.split(".")[:-1])

    pic.dump(G, open(stem + ".pkl", "wb"))

    newick = convert_network_to_newick_format(G)

    with open(out_fp, "w") as f:
        f.write(newick)

if __name__ == "__main__":
    main()
