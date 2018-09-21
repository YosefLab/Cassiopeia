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

    for d in depth.keys():

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

    net_fp = sys.argv[1]
    out_fp = sys.argv[2]

    G = pic.load(open(net_fp, "rb"))

    root = [n for n in G if G.in_degree(n) == 0][0]

    G  = set_depth(G, root)
    max_depth = get_max_depth(G)

    G = extend_dummy_branches(G, max_depth)

    G = set_progeny_size(G, root)

    ents = []
    for d in tqdm(range(1, max_depth), desc="Computing entropy at all depths"):
        ents.append(calc_entropy(G, depth=d))

    fig = plt.figure(figsize=(7,7))
    plt.bar(range(1, max_depth), ents)
    plt.title("Tree Entropy")
    plt.xlabel("Depth")
    plt.ylabel("Entropy")
    plt.savefig(out_fp)
