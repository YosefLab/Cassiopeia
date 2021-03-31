from ete3 import Tree
import networkx as nx
import pandas as pd
import numpy as np
from functools import reduce
from tqdm import tqdm
import matplotlib.pyplot as plt

from collections import OrderedDict, defaultdict

import scipy.stats as scs
import cassiopeia.TreeSolver.compute_meta_purity as cmp

from cassiopeia.Analysis import small_parsimony


def naive_fitch(t, meta):

    root = [n for n in t if t.in_degree(n) == 0][0]
    t = small_parsimony.assign_labels(t, meta)
    possible_labels = meta.unique()

    t = cmp.set_depth(t, root)

    label_to_j = dict(zip(possible_labels, range(len(possible_labels))))

    M = small_parsimony.draw_one_solution(t, possible_labels, label_to_j)

    M = pd.DataFrame(M)
    M.columns = possible_labels
    M.index = possible_labels

    return M


def fitch_count(t, meta):

    root = [n for n in t if t.in_degree(n) == 0][0]

    t = small_parsimony.assign_labels(t, meta)

    possible_labels = meta.unique()

    t = cmp.set_depth(t, root)
    t = small_parsimony.fitch_hartigan_bottom_up(t, root, possible_labels)

    bfs_postorder = [root]
    for e0, e1 in nx.bfs_edges(t, root):
        bfs_postorder.append(e1)

    node_to_i = dict(zip(bfs_postorder, range(len(t.nodes))))
    label_to_j = dict(zip(possible_labels, range(len(possible_labels))))

    L = small_parsimony._N(t, possible_labels, node_to_i, label_to_j)

    C = small_parsimony._C(t, L, possible_labels, node_to_i, label_to_j)

    M = pd.DataFrame(np.zeros((L.shape[1], L.shape[1])))
    M.columns = possible_labels
    M.index = possible_labels

    # count_mat: transitions are rows -> columns
    for s1 in possible_labels:
        for s2 in possible_labels:
            M.loc[s1, s2] = np.sum(
                C[node_to_i[root], :, label_to_j[s1], label_to_j[s2]]
            )

    return M


def assign_labels(tree, labels):

    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    for l in _leaves:
        tree.nodes[l]["label"] = [labels[l.name]]
    return tree


def shuffle_labels(meta):
    inds = meta.index.values
    np.random.shuffle(inds)
    meta.index = inds
    return meta


def plot_transition_probs(cout_arr, save_fp=None, title="", _order=None):

    # plot results
    np.fill_diagonal(count_arr.values, 0)

    mask = np.zeros_like(lg_to_countarr[7])
    np.fill_diagonal(mask, 1)

    count_arr = count_arr.apply(lambda x: x / max(1, x.sum()), axis=1)

    if _order:
        res = count_arr.loc[_order, _order]
    else:
        res = count_arr

    h = plt.figure(figsize=(10, 10))
    np.fill_diagonal(count_arr.values, np.nan)
    g = sns.heatmap(res, mask=mask, cmap="Reds", square=True)
    plt.ylabel("Origin Tissue")
    plt.xlabel("Destination Tissue")
    plt.title(title)
    g.set_facecolor("#bfbfbf")

    if save_fp:
        plt.savefig(save_fp)

    else:
        plt.show()
