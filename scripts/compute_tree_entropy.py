from __future__ import division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.setrecursionlimit(10000)

import pickle as pic
import networkx as nx
from pylab import *

import argparse

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

def get_max_depth(G):

    md = 0

    for n in G.nodes:
        
        if G.nodes[n]["depth"] > md:

            md = G.nodes[n]["depth"]

    return md

def extend_dummy_branches(G, max_depth):
    """
    Extends dummy branches from leaves to bottom of the tree for easier 
    calculations of entropy
    """

    leaves = [n for n in G.nodes if G.out_degree(n) == 0 and G.in_degree(n) == 1]
    for n in tqdm(leaves, desc="Extending dummy branches"):
        
        new_node_iter = 1
        while G.nodes[n]["depth"] < max_depth:

            d = G.nodes[n]["depth"]
            new_node = str(n) + "-" + str(new_node_iter) 
            parents = list(G.predecessors(n))
            for p in parents:
                G.remove_edge(p, n)
                G.add_edge(p, new_node)
            G.add_edge(new_node, n)
            
            G.nodes[new_node]["depth"] = d
            G.nodes[n]["depth"] = d + 1

            new_node_iter += 1 

    return G


def get_progeny_size(G, node):

    all_prog = [node for node in nx.dfs_preorder_nodes(G, node)]

    return len([n for n in all_prog if G.out_degree(n) == 0 and G.in_degree(n) == 1])

def set_depth(G, root):

    depth = nx.shortest_path_length(G, root)

    G.nodes[root]["depth"] = 0

    for d in tqdm(depth.keys(), desc='Setting depth'):

        G.nodes[d]["depth"] = depth[d]
    
    return G

def set_progeny_size(G, root):

    s = get_progeny_size(G, root)

    G.nodes[root]["prog_size"] = s

    for d in tqdm(G.nodes(), desc="Computing progeny size for each internal node"):

        s = get_progeny_size(G, d)
        G.nodes[d]["prog_size"] = s
    
    return G

def cut_tree(G, depth):

    nodes = []
    for n in G:

        if G.nodes[n]["depth"] == depth:
            nodes.append(n)

    return nodes

def calc_entropy(G, depth=0):

    nodes = cut_tree(G, depth)

    if len(nodes) == 0:
        return 0

    subclade_sizes = [G.nodes[n]["prog_size"] for n in nodes]

    if len(subclade_sizes) == 1:
        return 0

    tot = np.sum(subclade_sizes)
    probs = [s / float(tot) for s in subclade_sizes]

    return -np.sum(probs * np.log2(probs))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("netfp", type=str)
    parser.add_argument("--shuff", "-s", default="", type=str)
    parser.add_argument("out_fp", type=str)

    args = parser.parse_args()
    net_fp = args.netfp
    shuff_fp = args.shuff
    out_fp = args.out_fp

    G = pic.load(open(net_fp, "rb"))

    #make sure that all edges are collapsed
    #G = tree_collapse(G)

    root = [n for n in G if G.in_degree(n) == 0][0]

    G  = set_depth(G, root)
    max_depth = get_max_depth(G)

    G = extend_dummy_branches(G, max_depth)

    G = set_progeny_size(G, root)

    ents = []
    for d in tqdm(range(1, max_depth), desc="Computing entropy at all depths"):
        ents.append(calc_entropy(G, depth=d))

    if shuff_fp != "":

        print("Computing Statistics for Shuffled Data", end="\n\n")

        s_G = pic.load(open(shuff_fp, "rb"))
        root = [n for n in s_G if s_G.in_degree(n) == 0][0]

        s_G = set_depth(s_G, root)
        s_max_depth = get_max_depth(s_G)

        s_G = extend_dummy_branches(s_G, max_depth)
        s_G = set_progeny_size(s_G, root)

        s_ents = []
        for d in tqdm(range(1, s_max_depth), desc="Computing entropy at all depths for shuffled data"):
            s_ents.append(calc_entropy(s_G, depth=d))

        fig = plt.figure(figsize=(7,7))
        plt.plot(range(1, max_depth), ents, label="Reconstructed")
        plt.plot(range(1, s_max_depth), s_ents, label="Shuffled Reconstructed")
        plt.xlim(0, max(max_depth, s_max_depth))
        plt.ylim(0, max(max(s_ents), max(ents)) + 2)
        plt.xlabel('Depth')
        plt.ylabel('Entropy')
        plt.title("Tree Entropy")
        plt.legend()
        plt.savefig(out_fp)

    else:

        fig = plt.figure(figsize=(7,7))
        plt.bar(range(1, max_depth), ents)
        plt.title("Tree Entropy")
        plt.xlabel("Depth")
        plt.ylabel("Entropy")
        plt.savefig(out_fp)
