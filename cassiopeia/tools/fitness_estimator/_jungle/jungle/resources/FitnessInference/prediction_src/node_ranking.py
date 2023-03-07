#########################################################################################
#
# author: Richard Neher
# email: richard.neher@tuebingen.mpg.de
#
# Reference: Richard A. Neher, Colin A Russell, Boris I Shraiman.
#            "Predicting evolution from the shape of genealogical trees"
#
#########################################################################################
#
# node_ranking.py
# provides a class that runs the fitness inference on a given and ranks the nodes either
# by the mean posterior fitness or a few other metrics. it inherits fitness_inference.
# several utility functions to color trees by the resulting ranking.
#
#########################################################################################

import os
import random as pyrd
import time

import numpy as np
from scipy import stats

from .fitness_inference import *
from .tree_utils import *


##################################################
# SUBCLASS OF FITNESS INFERENCE WHICH RANKS THE NODES IN A TREE BY DIFFERENT METHODS
##################################################
class node_ranking(fitness_inference):
    """
    ranks the external and internal nodes of a tree using the fitness inference algorithm
    """

    def __init__(
        self,
        methods=[
            "mean_fitness",
            "polarizer",
            "branch_length",
            "depth",
            "expansion_score",
        ],
        time_bins=None,
        pseudo_count=5,
        *args,
        **kwargs
    ):
        """
        no required arguments
        keyword arguments
        methods           --   methods used to rank the nodes
        eps_branch_length --   argument to fitness_inference: minimal branch_length
        time_bins         --   temporal bins used to calculate the expansion score
        pseudo_count      --   pseudo count used for calculating frequencies in time_bins
        """

        fitness_inference.__init__(self, *args, **kwargs)
        self.methods = methods
        if time_bins is not None:
            self.time_bins = sorted(time_bins)
        else:
            self.time_bins = None
        self.pseudo_count = pseudo_count

    def compute_rankings(self):
        """
        infer ancestral fitness and evaluate the scores for each method provided
        """
        if "mean_fitness" in self.methods:
            if verbose:
                tmp_t = time.time()
                print("node_ranking: inferring ancestral fitness...", end=" ")
            self.infer_ancestral_fitness()
            if verbose:
                print("done in ", np.round(time.time() - tmp_t, 2), "s")
        if "polarizer" in self.methods:
            self.calculate_polarizers()

        if verbose:
            tmp_t = time.time()
            print("node_ranking: calculating alternative rankings...", end=" ")

        if "depth" in self.methods or (
            ("expansion_score" in self.methods) and (self.time_bins is not None)
        ):
            for c in self.terminals + self.non_terminals:
                c.depth = self.depths[c]
        if ("expansion_score" in self.methods) and (self.time_bins is not None):
            self.expansion_score()

        for method in self.methods:
            if callable(method):
                method(self.terminals, self.non_terminals)
        if verbose:
            print("done in ", np.round(time.time() - tmp_t, 2), "s")

    def expansion_score(self):
        """
        calculates the fraction of all leafs under a node and regresses
        on the log of that fraction. slope is saved as .expansion_score
        in each internal node
        """
        if not self.time_bins:
            print(
                "expansion_score(): need time bins to calculate expansion score"
            )
            return

        self.leafs_by_bin = self.sort_leafs_in_time_bins()
        total_counts_by_bin = np.array(
            [len(leaflist) for leaflist in self.leafs_by_bin], dtype=float
        )
        for node in self.non_terminals:
            node.counts_by_bin = np.array(
                [
                    len(leaflist)
                    for leaflist in self.sort_leafs_in_time_bins(node)
                ]
            )
            node.temporal_frequency = (
                node.counts_by_bin + self.pseudo_count
            ) / (total_counts_by_bin + self.pseudo_count)
            if len(self.time_bins) > 1:
                node.expansion_score = stats.linregress(
                    np.arange(len(self.time_bins)),
                    np.log(node.temporal_frequency),
                )[0]

    def sort_leafs_in_time_bins(self, node=None):
        """
        makes a list of nodes belonging to each time bin
        """
        if not self.time_bins:
            print("sort_leafs_in_time_bins: no time bins specifies")
            return

        # if no parent node specified, use all terminals
        if node:
            tmp_terminals = node.get_terminals()
        else:
            tmp_terminals = self.terminals

        tmp_leafs_by_bin = [[] for b in self.time_bins]
        for c in tmp_terminals:
            tmp_bin_array = [c.depth < tau for tau in self.time_bins]
            if True in tmp_bin_array:
                tmp_leafs_by_bin[tmp_bin_array.index(True)].append(c)

        return tmp_leafs_by_bin

    def rank_by_method(self, nodes=None, method="mean_fitness", scramble=True):
        """
        keyword arguments:
        nodes    -- list of nodes to be sorted
        method   -- attribute by which to sort, needs to be present in each node
        scramble -- scramble the nodes before ordering to remove previous order
        """
        if nodes is None:
            nodes = self.terminals
        # sort the nodes according the desired ranking
        if scramble:
            from random import shuffle as rd_shuffle

            rd_shuffle(nodes)
        nodes.sort(reverse=True, key=lambda x: x.__getattribute__(method))
        for ni, node in enumerate(nodes):
            node.rank = ni + 1
        return nodes

    def correlation_between_scores(self, score1, score2, node_set=None):
        """
        calculate the rank correlation between different node attributes
        """
        if node_set is None:
            node_set = self.terminals
        try:
            values1 = [t.__getattribute__(score1) for t in node_set]
            values2 = [t.__getattribute__(score2) for t in node_set]
            return stats.spearmanr(values1, values2)
        except AttributeError as e:
            print(e)
            return None
        except:
            print("correlation_between_scores: error ")
            return None

    def ranking_quality(self):
        """
        return the rank of the lowest terminal node less than one std away from the top
        """
        self.terminals.sort(reverse=True, key=lambda x: x.mean_fitness)
        cut_off = self.terminals[0].mean_fitness - np.sqrt(
            self.terminals[0].var_fitness
        )
        ci = 1
        while self.terminals[ci].mean_fitness > cut_off:
            ci += 1
        return ci

    def color_tree(
        self,
        nodes=None,
        method="mean_fitness",
        offset=0.000,
        n_labels=10,
        scramble=True,
        cmap=None,
    ):
        """
        use a colormap to display the ranking of the leafs
        keyword arguments:
        nodes    -- list of nodes to be sorted
        method   -- attribute by which to sort, needs to be present in each node
        offset   -- extend terminal branches by this amount to increase visibility
        n_labels -- number of nodes for which a label indicating rank is produced
        scramble -- scramble the nodes before ordering to remove previous order
        """
        if verbose:
            tmp_time = time.time()
            print("color_tree:...", end=" ")
        if cmap is None:
            from matplotlib import cm

            cmap = cm.jet
        nodes = self.rank_by_method(nodes, method, scramble)
        n_nodes = len(nodes)
        for ci, c in enumerate(nodes):
            # set color. this needs explicit conversion to int
            c.color = [
                int(x * 240)
                for x in np.array(
                    cmap(int(255.0 * (n_nodes - ci) ** 1 / n_nodes**1))[:3]
                )
            ]
            if c.is_terminal():
                c.branch_length += offset
            if ci < n_labels:
                c.rank_label = " " * ci + str(c.rank)
            else:
                c.rank_label = None
        if verbose:
            print("done in", np.round(time.time() - tmp_time))

    def interpolate_color(self, tree=None):
        """
        color internal nodes with the mean of leaf color. this will end up
        being grey in most cases
        """
        if tree is None:
            tree = self.T
        for c in tree.get_nonterminals():
            tmp_color = np.array([0, 0, 0])
            terms = c.get_terminals()
            leaf_count = 0
            for leaf in terms:
                if leaf.color:
                    tmp_color += np.array(
                        [leaf.color.red, leaf.color.green, leaf.color.blue]
                    )
                    leaf_count += 1
            if leaf_count:
                c.color = (tmp_color / leaf_count).tolist()
            else:
                c.color = (0, 0, 0)

    def color_other_tree(
        self,
        nodes,
        method="mean_fitness",
        offset=0.000,
        n_labels=10,
        scramble=True,
        cmap=None,
    ):
        """
        transfer the score to the nodes of another trees (matched by name)
        call color_tree with all keyword arguments and the list of nodes
        """
        look_up = {c.name: c for c in self.terminals + self.non_terminals}
        nodes_to_color = []
        for c in nodes:
            if c.name in look_up:
                c.__setattr__(method, look_up[c.name].__getattribute__(method))
                nodes_to_color.append(c)
            else:
                c.color = (100, 100, 100)
                c.rank_label = None
        self.color_tree(
            nodes_to_color,
            method,
            offset,
            n_labels,
            scramble=scramble,
            cmap=cmap,
        )

    def best_node(self, method="mean_fitness", nodes=None):
        """
        return the best among a set of nodes
        """
        if nodes is None:
            nodes = self.terminals
        self.rank_by_method(nodes, method, scramble=True)
        if nodes[0].name is None:
            nodes[0].name = (
                "best by method "
                + method
                + " among #of nodes "
                + str(len(nodes))
            )
        return nodes[0]

    def rank_labels(self, node):
        try:
            return node.rank_label
        except:
            return None
