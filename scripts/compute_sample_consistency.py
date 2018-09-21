from __future__ import division
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.setrecursionlimit(10000)
import pickle as pic

import networkx as nx
from collections import defaultdict
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
    for n in leaves:
        
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

def set_progeny_size(G, root):

    s = get_progeny_size(G, root)

    G.nodes[root]["prog_size"] = s

    for d in tqdm(G.nodes(), desc="Computing progeny size for each internal node"):

        s = get_progeny_size(G, d)
        G.nodes[d]["prog_size"] = s
    
    return G

def get_progeny_size(G, node):

    all_prog = [node for node in nx.dfs_preorder_nodes(G, node)]

    return len([n for n in all_prog if G.out_degree(n) == 0 and G.in_degree(n) == 1])

def get_children_of_clade(G, node):

    all_prog = [node for node in nx.dfs_preorder_nodes(G, node)]
    return [n for n in all_prog if G.out_degree(n) == 0 and G.in_degree(n) == 1]
         
def get_meta_counts(G, node, metavals):
    
    meta_counts = defaultdict(dict)    
    children_vals = [G.nodes[n]["meta"] for n in get_children_of_clade(G, node)]
    for m in metavals:
    
        meta_counts[m] = children_vals.count(m)
    
    return meta_counts


def set_depth(G, root):

    depth = nx.shortest_path_length(G, root)

    for d in depth.keys():

        G.nodes[d]["depth"] = depth[d]
    
    return G

def cut_tree(G, depth):

    nodes = []
    for n in G.nodes:

        if G.nodes[n]["depth"] == depth:
            nodes.append(n)

    return nodes

def calc_entropy(G, metavals, depth=0):

    nodes = cut_tree(G, depth)

    if len(nodes) == 0:
        return 1.0

    # metacounts is a list of dictionaries, each tallying the number of
    # occurrences of each meta value in the subclade below the node n
    metacounts = dict(zip(nodes, [get_meta_counts(G, n, metavals) for n in nodes]))

    ents = []
    for mkey in metacounts.keys():
        
        mc = metacounts[mkey]

        if len(mc.keys()) == 1:
            return 1.0

        tot = np.sum(list(mc.values()))
        probs = [s / float(tot) for s in list(mc.values()) if s != 0]

        ei = -np.sum(probs * np.log2(probs))
        pi = G.nodes[mkey]["prog_size"]

        ents.append(ei)

    return ents

def assign_meta(G, meta):

    root = [node for node in G.nodes() if G.in_degree(node) == 0][0]

    leaves = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
    metadict = {}

    for l in leaves:
        G.nodes[l]['meta'] = meta.loc[l] 

    return G

if __name__ == "__main__":

    netfp = sys.argv[1]
    meta_fp = sys.argv[2]
    out_fp = sys.argv[3]
    rand = "" if len(sys.argv) < 5 else sys.argv[4]

    show_random = False
    if rand == "--random":
        show_random = True

    out_fp_stem = "".join(out_fp.split(".")[:-1])

    meta = pd.read_csv(meta_fp, sep='\t', index_col = 0)

    G = pic.load(open(netfp, "rb"))

    root = [n for n in G if G.in_degree(n) == 0][0]

    G = set_depth(G, root)
    max_depth = get_max_depth(G)
    G = extend_dummy_branches(G, max_depth)

    G = set_progeny_size(G, root)

    for i in tqdm(meta.columns, desc="Processing each meta item"):
        meta_vals = list(meta[i].unique())
        G = assign_meta(G, meta[i])

        ents = []
        for d in tqdm(range(1, max_depth), desc="Calculating entropy at each level"):

            ents.append(np.mean(calc_entropy(G, meta_vals, depth=d)))

        if show_random:
            ents_rand = []
            meta_rand = meta[i]
            meta_rand.index = np.random.permutation(meta_rand.index)
            G = assign_meta(G, meta_rand)
            for d in tqdm(range(1, max_depth), desc="Calculating entropy for permuted " + str(i)):
                ents_rand.append(np.mean(calc_entropy(G, meta_vals, depth=d)))
            
            width = 0.35
            plt.bar(np.arange(1, max_depth)-width/2, ents_rand, width, label="Random")
            plt.bar(np.arange(1, max_depth)+width/2,ents, width, label="True")
            plt.ylabel("Meta Entropy")
            plt.xlabel("Depth")
            plt.title("Meta Entropy, " + str(i))
            plt.legend()
            plt.savefig(out_fp_stem + "_" + str(i) + ".png")
       
        else:
            fig = plt.figure(figsize=(7,7))
            plt.bar(range(1, max_depth), ents)
            plt.title("Meta Entropy, " + str(i))
            plt.xlabel("Depth")
            plt.ylabel("Meta Entropy")
            plt.savefig(out_fp_stem + "_" + str(i) + ".png")

    
        
