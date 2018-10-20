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

    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
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

def calc_entropy(G, metavals, meta_freqs, depth=0):

    nodes = cut_tree(G, depth)

    if len(nodes) == 0:
        return 1.0

    # metacounts is a list of dictionaries, each tallying the number of
    # occurrences of each meta value in the subclade below the node n
    metacounts = dict(zip(nodes, [get_meta_counts(G, n, metavals) for n in nodes]))

    num_leaves = sum([G.nodes[m]["prog_size"] for m in metacounts.keys()])

    ents = []
    exp_ents = []
    chi_sq = []
    pvals = []
    for mkey in metacounts.keys():

        mc = metacounts[mkey]

        if len(mc.keys()) == 1:
            return 1.0

        tot = np.sum(list(mc.values()))
        probs = dict(zip(mc.keys(), [s / float(tot) for s in list(mc.values())]))
        nnz_probs = np.array([p for p in probs.values() if p > 0])

        ni = G.nodes[mkey]["prog_size"]
        ei = -np.sum(nnz_probs * np.log2(nnz_probs))
        pi = ni / num_leaves

        exp = dict(zip(meta_freqs.keys(), [ni * meta_freqs[m] for m in meta_freqs.keys()])) 
        exp_probs = dict(zip(exp.keys(), [s / float(tot) for s in exp.values()]))

        exp_ei = -np.sum(list(exp_probs.values()) * np.log2(list(exp_probs.values())))
        exp_ents.append(exp_ei * pi)

        tstat = np.sum( [ (exp[m] - mc[m])**2 / exp[m] for m in meta_freqs.keys()])  
        chi_sq.append(tstat)

        pval = 1 - stats.chi2.cdf(tstat, len(meta_freqs.keys()) - 1)
        pvals.append(pval)

        ents.append(ei * pi)

    return ents, exp_ents, chi_sq, pvals

def assign_meta(G, meta):

    root = [node for node in G.nodes() if G.in_degree(node) == 0][0]

    leaves = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
    metadict = {}

    for l in leaves:
        G.nodes[l]['meta'] = meta.loc[l]

    return G

def add_redundant_leaves(G, cm):
    """
    To fairly take into account sample purity, we'll add back in 'redundant' leaves (i.e.
    leaves that were removed because of non-unique character strings).
    """

    # create lookup value for duplicates
    cm["lookup"] = cm.astype('str').apply(''.join, axis=1)
    net_nodes = np.intersect1d(cm.index, [n for n in G])

    uniq = cm.loc[net_nodes]

    # find all non-unique character states in cm
    nonuniq = np.setdiff1d(cm.index, np.array([n for n in G]))

    for n in nonuniq:

        new_node = str(n)
        _leaf = uniq.index[uniq["lookup"] == cm.loc[n]["lookup"]][0]
        parents = list(G.predecessors(_leaf))
        for p in parents:
            G.add_edge(p, new_node)

        G.nodes[new_node]["depth"] = G.nodes[_leaf]["depth"]

    return G

def calculate_empirical_pvalues(real, rand_ent_dist):
    """
    Calculate empirical p value from generated random entropy distribution

    """

    pvs = []

    for i in range(len(real)):

        obs = real[i]
        dist = rand_ent_dist[i]

        # want to ask how many times we observed less entropy (more sample purity) in the
        # random distribution than our observed purity from some algorithm
        pv = (1 + sum(dist < obs)) / (len(dist) + 1) # apply bias correction

        pvs.append(pv)

    return np.array(pvs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("netfp", type=str)
    parser.add_argument("meta_fp", type=str)
    parser.add_argument("char_fp", type=str)
    parser.add_argument("out_fp", type=str)
    parser.add_argument("--shuff", "-s", default="", type=str)

    args = parser.parse_args()
    netfp = args.netfp
    meta_fp = args.meta_fp
    char_fp = args.char_fp
    out_fp = args.out_fp
    shuff_fp = args.shuff
    permute = False

    out_fp_stem = "".join(out_fp.split(".")[:-1])

    meta = pd.read_csv(meta_fp, sep='\t', index_col = 0)

    cm = pd.read_csv(char_fp, sep='\t', index_col = 0)

    G = pic.load(open(netfp, "rb"))

    root = [n for n in G if G.in_degree(n) == 0][0]

    G = set_depth(G, root)
    max_depth = get_max_depth(G)
    G = extend_dummy_branches(G, max_depth)

    # make sure that extend dummy branches worked 
    leaves = [n for n in G if G.out_degree(n) == 0]
    assert (False not in [max_depth == G.nodes[l]['depth'] for l in leaves])

    #G = add_redundant_leaves(G, cm)

    G = set_progeny_size(G, root)

    for i in tqdm(meta.columns, desc="Processing each meta item"):
        meta_vals = list(meta[i].unique())
        G = assign_meta(G, meta[i])
        
        meta_freqs = defaultdict(int)
        tree_vals = [G.nodes[n]["meta"] for n in get_children_of_clade(G, root)]
        for m in meta_vals:
            meta_freqs[m] = tree_vals.count(m) / len(tree_vals)

        ents = []
        exp_ents = []
        tstats = []
        pvalues = []
        for d in tqdm(range(1, max_depth), desc="Calculating entropy at each level"):

            obs_ents, e_ents, chi_sq, pvals = calc_entropy(G, meta_vals, meta_freqs, depth=d)
            ents.append(np.mean(obs_ents))
            exp_ents.append(np.mean(e_ents))
            tstats.append(np.mean(chi_sq))
            pvalues.append(np.mean(pvals))
            #pvalues.append(1 - stats.chi2.cdf(np.mean(chi_sq), len(meta_vals) - 1))
    
        if shuff_fp != "":

            print("Computing Statistics for Shuffled Data", end="\n\n")

            s_G = pic.load(open(shuff_fp, "rb"))
            s_G = assign_meta(s_G, meta[i])
            root = [n for n in s_G if s_G.in_degree(n) == 0][0]

            s_G = set_depth(s_G, root)
            s_max_depth = get_max_depth(s_G)

            s_G = extend_dummy_branches(s_G, max_depth)
            s_G = set_progeny_size(s_G, root)

            s_ents = []
            s_exp_ents = []
            s_tstats = []
            s_pvalues = []
            for d in tqdm(range(1, max_depth), desc="Calculating entropy at each level"):
    
                obs_ents, e_ents, chi_sq, pvals = calc_entropy(s_G, meta_vals, meta_freqs, depth=d)
                s_ents.append(np.mean(obs_ents))
                s_exp_ents.append(np.mean(e_ents))
                s_tstats.append(np.mean(chi_sq))
                #s_pvalues.append(1 - stats.chi2.cdf(np.mean(chi_sq), len(meta_vals) - 1))
                s_pvalues.append(np.mean(pvals))

            fig = plt.figure(figsize=(7,7))
            plt.plot(range(1, max_depth), ents, label="Reconstructed")
            plt.plot(range(1, s_max_depth), s_ents, label="Shuffled Reconstructed")
            plt.xlim(0, max(max_depth, s_max_depth))
            plt.ylim(0, max(max(s_ents), max(ents)))
            plt.xlabel('Depth')
            plt.ylabel('Meta Entropy')
            plt.legend()
            plt.title("Meta Entropy, " + str(i))
            plt.savefig(out_fp_stem + "_" + str(i) + '.png')

            fig = plt.figure(figsize=(7, 7))
            plt.plot(np.arange(1, max_depth), -1*np.log10(pvalues), label="Reconstructed")
            plt.plot(np.arange(1, s_max_depth), -1*np.log10(s_pvalues), label="Shuffled Reconstructed")
            plt.ylabel("- log(P Value)")
            plt.xlabel("Depth")
            plt.title("Significance of Sample Purity vs Depth, " + str(i))
            plt.legend()
            plt.savefig(out_fp_stem + "_significance_" + str(i) + ".png")
            plt.close()
        
        else:

            fig = plt.figure(figsize=(7, 7))
            plt.plot(np.arange(1, max_depth), -1*np.log10(pvalues))
            plt.ylabel("- log(P Value)")
            plt.xlabel("Depth")
            plt.title("Significance of Sample Purity vs Depth, " + str(i))
            plt.savefig(out_fp_stem + "_significance_" + str(i) + ".png")
            plt.close()

            fig = plt.figure(figsize=(7, 7))
            plt.plot(np.arange(1, max_depth), ents, label='True')
            plt.plot(np.arange(1, max_depth), exp_ents, label="Chi-Squared Null")
            plt.ylabel("Meta Entropy")
            plt.xlabel("Depth")
            plt.title("Meta Entropy, " + str(i))
            plt.legend()
            plt.savefig(out_fp_stem + "_" + str(i) + ".png")
            plt.close()

        fig = plt.figure(figsize=(7, 7))
        plt.plot(np.arange(1, max_depth), tstats)
        plt.xlabel("Depth")
        plt.ylabel("Mean Chi Squared Stat")
        plt.title("Mean Chi Sq Statistic Per Depth")
        plt.savefig(out_fp_stem + "_chisq_" + str(i) + ".png")
