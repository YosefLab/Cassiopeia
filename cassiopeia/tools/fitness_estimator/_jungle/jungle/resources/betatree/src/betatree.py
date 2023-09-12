"""
author:     Taylor Kessinger & Richard Neher
date:       10/07/2014
content:    generate beta coalescent trees and calculate their SFS
"""
import random as rand

import numpy as np
import scipy.special as sf
from Bio import Phylo


class betatree(object):
    """
    class that simulates a beta coalescent tree
    parameters:
    sample_size -- number of leaves
    alpha       -- parameter of the merger distribution 2 for Kingman, 1 for BSC
    """

    def __init__(self, sample_size, alpha=2):
        self.alpha = alpha
        self.n = sample_size
        # auxillary arrays
        self.k = np.arange(self.n + 1)
        if alpha == 1:
            self.inv_k = 1.0 / np.arange(1, self.n + 2)
            self.inv_kkp1 = self.inv_k / np.arange(2, self.n + 3)
            self.cum_sum_inv_kkp1 = np.array(
                [0] + np.cumsum(self.inv_kkp1).tolist()
            )
        elif alpha > 1 and alpha < 2:
            self.normalizer = 1.0 / sf.gamma(alpha) / sf.gamma(2 - alpha)
            self.gamma_ratiom = np.exp(
                sf.gammaln(self.k - self.alpha) - sf.gammaln(self.k + 1)
            )
            self.gamma_ratiop = np.exp(
                sf.gammaln(self.k + self.alpha) - sf.gammaln(self.k + 1)
            )

    def init_tree(self):
        """
        constructs the blocks that are to be merged, each leave corresponds
        to a BioPython clade object
        """
        self.blocks = [
            Phylo.BaseTree.Clade(name=str(i), branch_length=0)
            for i in range(self.n)
        ]

    def coalescence_event(self):
        """
        choose the time of the next merger,the number of blocks to merge,
        and perform the merger
        """
        merger_size = self.whichp(len(self.blocks))
        waiting_time = self.waiting_time()
        # branch length is first accumulated. once the tree is done, take
        # differentials of parent and child branchlength to get the actual
        # branchlength
        for clade in self.blocks:
            clade.branch_length += waiting_time

        # randomly pick some blocks (there are (p choose k) possibilities)
        merging_blocks = rand.sample(self.k[: len(self.blocks)], merger_size)
        self.merge_clades(merging_blocks)

    def merge_clades(self, merging_blocks):
        """
        creates a new clade whose children are the merging blocks
        """
        # instantiate at Clade object with children given by the merging blocks
        new_clade = Phylo.BaseTree.Clade(
            clades=[self.blocks[i] for i in merging_blocks]
        )
        # set the branch length to that of the children
        new_clade.branch_length = self.blocks[merging_blocks[0]].branch_length
        # remove the merging blocks from the active blocks
        for i in sorted(merging_blocks, reverse=True):
            self.blocks.pop(i)
        self.blocks.append(new_clade)

    def clean_up_subtree(self, clade):
        """
        calculate the branch length and number of children for each node
        """
        if clade.is_terminal():
            clade.weight = 1
            return
        else:
            clade.weight = 0
            clade.branch_length -= clade.clades[0].branch_length
            for child in clade.clades:
                self.clean_up_subtree(child)
                clade.weight += child.weight
            return

    def waiting_time(self):
        """
        returns the waiting time to the next merger.
        """
        b = len(self.blocks)
        if self.alpha == 1:  # the BSC merger rate
            # the BSC merger rate is simply b-1 = 1 / (sum_k 1/k(k-1))
            dt = np.random.exponential(1.0 / (b - 1))
        elif self.alpha == 2:  # the Kingman merger rate
            dt = np.random.exponential(2.0 / b / (b - 1))
        else:  # the general beta coalescent case
            rate = (
                b
                * (
                    self.gamma_ratiom[2 : b + 1]
                    * self.gamma_ratiop[b - 2 :: -1]
                ).sum()
                * self.normalizer
            )
            dt = np.random.exponential(
                1.0 / rate
            )  # this product gives the Beta coalescent merger rate
        return dt

    def whichp(self, b):
        """
        generates the merger size distribution, then samples from it.
        parameters:
            b: the number of extant lineages.
        """
        if self.alpha == 1:  # BSC case
            # merger degree is distributed as 1/(k-1)/k, pull a random number from
            # the range [0,\sum_k<=b  1/(k-1)/k. The cum_sum_kkp1 array is shifted by 2, hence the b-1
            randvar = np.random.uniform(0, self.cum_sum_inv_kkp1[b - 1])
            # determine the maximal k such that the sum is smaller than rand
            return np.where(self.cum_sum_inv_kkp1[:b] > randvar)[0][0] + 1
        elif self.alpha == 2:  # Kingman case
            return 2
        else:  # other Beta coalescents
            # calculate the cumulative distribution of the variable part of the merger rate
            # normalizer and b dependent prefactors omitted
            cum_rates = np.cumsum(
                self.gamma_ratiom[2 : b + 1] * self.gamma_ratiop[b - 2 :: -1]
            )
            randvar = np.random.uniform(0, cum_rates[-1])
            return np.where(cum_rates > randvar)[0][0] + 2

    def coalesce(self):
        """
        simulates the Beta coalescent process for arbitrary alpha.
        parameters:
            K0: the initial population size.
            alpha: parameter for the Beta coalescent. set to 2 for Kingman and 1 for Bolthausen-Sznitman.
        """

        self.init_tree()
        # while the whole tree is not merged yet
        while len(self.blocks) != 1:
            self.coalescence_event()

        self.clean_up_subtree(self.blocks[0])
        self.BioTree = Phylo.BaseTree.Tree(root=self.blocks[0])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()

    # alpha=2 -> Kingman coalescent tree
    myT = betatree(100, alpha=2)
    myT.coalesce()
    myT.BioTree.ladderize()
    Phylo.draw(myT.BioTree, label_func=lambda x: None)
    plt.title("Kingman: alpha=2")
    plt.savefig("example_trees/kingman.pdf")

    # alpha=1 -> Bolthausen-Sznitman coalescent tree
    myT = betatree(100, alpha=1)
    myT.coalesce()
    myT.BioTree.ladderize()
    Phylo.draw(myT.BioTree, label_func=lambda x: None)
    plt.title("Bolthausen-Sznitman: alpha=1")
    plt.savefig("example_trees/bolthausen_sznitman.pdf")

    # alpha=1.5 -> general beta coalescent tree
    myT = betatree(100, 1.5)
    myT.coalesce()
    myT.BioTree.ladderize()
    Phylo.draw(myT.BioTree, label_func=lambda x: None)
    plt.title("alpha=1.5")
    plt.savefig("example_trees/alpha_1.5.pdf")
