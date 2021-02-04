from ete3 import Tree
import networkx as nx
import pandas as pd
import numpy as np
from functools import reduce
import itertools

from collections import defaultdict

import cassiopeia.TreeSolver.compute_meta_purity as cmp
from networkx.algorithms.traversal.depth_first_search import dfs_tree

def reconcile_S1(T):
    """
    Prune the Fitch-Hartigan S1 arrays across T to respect only solutions that are guaranteed to 
    be optimal in the subtree below any internal node and legal with respect to its parent's 
    set of potential states.
    """

    source = [n for n in T if T.in_degree(n) == 0][0]
    for e in nx.dfs_edges(T):

        p, c = e[0], e[1]
        ns = np.intersect1d(T.nodes[p]["S1"], T.nodes[c]["S1"]).tolist()

        if len(ns) > 0 and len(ns) == len(T.nodes[p]["S1"]):
            T.nodes[c]["S1"] = ns
        else:
            T.nodes[c]["S1"] = list(T.nodes[c]["S1"])
    return T

def draw_one_solution(T, possible_labels, label_to_j):

    M = len(possible_labels)
    C = np.zeros((M, M))
    root = [n for n in T if T.in_degree(n) == 0][0]
    
    T = fitch_hartigan(T)
    
    # now count transitions
    for v in nx.dfs_postorder_nodes(T, source=root):

        v_lab = T.nodes[v]['label']
        i = label_to_j[v_lab]

        children = list(T.successors(v))
        for c in children:
            
            c_lab = T.nodes[c]['label']
            j = label_to_j[c_lab]

            C[i, j] += 1

    return C

def _N(T, possible_assignments, node_to_i, label_to_j):
    def fill_DP(v, s):

        if T.out_degree(v) == 0:
            return 1
        
        children = list(T.successors(v))
        A = np.zeros((len(children)))
        
        legal_states = []

        for i, u in zip(range(len(children)), children):
            
            if s not in T.nodes[u]['S1']:
                legal_states = T.nodes[u]['S1']
            else:
                legal_states = [s]
                
            A[i] = np.sum([L[node_to_i[u], label_to_j[sp]] for sp in legal_states])

        return np.prod([A[u] for u in range(len(A))])

    L = np.full((len(T.nodes), len(possible_assignments)), 0.0)

    root = [n for n in T if T.in_degree(n) == 0][0]

    for n in nx.dfs_postorder_nodes(T, source=root):
        for s in T.nodes[n]['S1']:
            L[node_to_i[n], label_to_j[s]] = fill_DP(n, s)


    return L

            
            
def _C(
    T, L, possible_labels, node_to_i, label_to_j,
):
    def fill_transition_DP(v, s, s1, s2, obs_transitions):

        if T.out_degree(v) == 0:
            return 0

        children = list(T.successors(v))
        A = np.zeros((len(children)))
        LS = [[]] * len(children)

        for i, u in zip(range(len(children)), children):
            LS_u = None
            if s in T.nodes[u]["S1"]:
                LS[i] = [s]
            else:
                LS[i] = T.nodes[u]["S1"]

            A[i] = np.sum([C[node_to_i[u], label_to_j[sp], label_to_j[s1], label_to_j[s2]] for sp in LS[i]])

            if s1 == s and s2 in LS[i]:
                A[i] += L[node_to_i[u], label_to_j[s2]]

        parts = []
        for i, u in zip(range(len(children)), children):
            prod = 1

            for k, up in zip(range(len(children)), children):
                fact = 0
                if up == u:
                    continue
                for sp in LS[k]:
                    fact += L[node_to_i[up], label_to_j[sp]]

                prod *= fact

            part = A[i] * prod
    
            parts.append(part)

        return np.sum(parts)

    obs_transitions = defaultdict(list)
    C = np.zeros((len(T.nodes), L.shape[1], L.shape[1], L.shape[1]))
    root = [n for n in T if T.in_degree(n) == 0][0]

    for n in nx.dfs_postorder_nodes(T, source=root):
        for s in T.nodes[n]["S1"]:
            for s_pair in itertools.product(possible_labels, repeat=2):
                s1, s2 = s_pair[0], s_pair[1]
                C[
                    node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]
                ] = fill_transition_DP(n, s, s1, s2, obs_transitions)

    return C

def fitch_hartigan_bottom_up(tree, root, S):
    # run Haritigan's bottom up phase on an input tree with a specified root and alphabet of internal nodes
    # stored in S

    md = cmp.get_max_depth(tree, root)

    # bottom up approach
    d = md
    while d >= 0:

        internal_nodes = cmp.cut_tree(tree, d)
        for i in internal_nodes:
            children = list(tree.successors(i))

            if len(children) == 1:
                tree.nodes[i]["S1"] = tree.nodes[children[0]]["S1"]
                tree.nodes[i]['S2'] = []
                continue
            if len(children) == 0:
                if "S1" not in tree.nodes[i].keys():
                    raise Exception("This should have a label!")
                continue
            
            all_labels = np.concatenate(([tree.nodes[c]['S1'] for c in children]))
            
            freqs = []
            for k in S:
                freqs.append(np.count_nonzero(all_labels == k))
            
            S1 = S[np.where(freqs == np.max(freqs))]
            S2 = S[np.where(freqs == (np.max(freqs) - 1))]
            
            tree.nodes[i]['S1'] = S1
            tree.nodes[i]['S2'] = S2

        d -= 1

    return tree


def fitch_hartigan_top_down(tree, root):
    # Run Hartigan's top-down refinement, selecting one optimal solution from tree rooted at a 
    # defined root.

    md = cmp.get_max_depth(tree, root)

    # Phase 2: top down assignment
    tree.nodes[root]["label"] = np.random.choice(tree.nodes[root]["S1"])
    d = 1
    while d <= md:

        internal_nodes = list(cmp.cut_tree(tree, d))

        for i in internal_nodes:

            parent = list(tree.predecessors(i))[0]
            
            if tree.nodes[parent]['label'] in tree.nodes[i]['S1']:
                tree.nodes[i]['label'] = tree.nodes[parent]['label']
                
            elif tree.nodes[parent]['label'] in tree.nodes[i]['S2']:
                
                choices = tree.nodes[i]['S1']
                # choices = np.union1d(tree.nodes[parent]['label'], tree.nodes[i]['S1'])
                tree.nodes[i]['label'] = np.random.choice(choices)

            else:
                tree.nodes[i]['label'] = np.random.choice(tree.nodes[i]['S1'])

        d += 1

    return tree


def fitch_hartigan(tree):
    """
    Runs the Hartigan algorithm (a generalization to Fitch's algorithm on nonbinary trees)
    on tree given the labels for each leaf. Returns the tree with labels on internal node.
    """

    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    
    # form candidate set of labels for each internal node
    S = np.unique(np.concatenate([tree.nodes[l]['S1'] for l in _leaves]))

    tree = cmp.set_depth(tree, root)
    tree = fitch_hartigan_bottom_up(tree, root, S)

    tree = fitch_hartigan_top_down(tree, root)

    return tree

def assign_labels(tree, labels):

    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    for l in _leaves:
        tree.nodes[l]["S1"] = [labels.loc[l.name]]
        tree.nodes[l]["S2"] = []
        
    return tree

def score_parsimony(tree):

    score = 0
    for e in tree.edges():
        source = e[0]
        dest = e[1]

        if "label" not in tree.nodes[source] or "label" not in tree.nodes[dest]:
            raise Exception("Internal Nodes are not labeled - run fitch_hartigan first")

        if tree.nodes[source]["label"] != tree.nodes[dest]["label"]:
            score += 1

    return score


def score_parsimony_cell(tree, root, cell_label):

    score = 0

    path = nx.shortest_path(tree, root, cell_label)

    i = 0
    while i < len(path) - 1:

        source = path[i]
        dest = path[i + 1]

        if "label" not in tree.nodes[source] or "label" not in tree.nodes[dest]:
            raise Exception("Internal Nodes are not labeled - run fitch_hartigan first")

        if tree.nodes[source]["label"] != tree.nodes[dest]["label"]:
            score += 1

        i += 1

    return score
