import copy
import numpy as np


class SFS:

    EULERS_CONSTANT = 0.57721566490153286060

    def __init__(self, counts, T=None):
        self.counts = counts
        self.T = T
        self.binned = None
        self._fay_and_wus_H = None
        self._zengs_E = None
        self._tajimas_D = None
        self._ferrettis_L = None

    @classmethod
    def from_tree(cls, T):
        """ Construct site frequency spectrum from tree """
        counts = np.zeros(len(T)+1)
        for clade in T.iter_descendants():
            counts[len(clade)] += clade.dist
        return SFS(counts, T)

    def bin(self, bins, n_bins=None, min_edge=None, max_edge=None):
        """ Bin site frequency spectrum """

        # If bins are not explicitly passed, set bins based on keyword (linear, log, or logit)
        if bins == "linear":
            if n_bins is None:
                n_bins = 20
            if min_edge is None:
                min_edge = 0
            if max_edge is None:
                max_edge = 1
            self.bin_edges = np.linspace(min_edge, max_edge, n_bins)
            self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1])/2
        elif bins == "log":
            if n_bins is None:
                n_bins = 20
            if min_edge is None:
                min_edge = -5
            if max_edge is None:
                max_edge = 0
            self.bin_edges = np.logspace(min_edge, max_edge, n_bins)
            self.bin_centers = np.sqrt(self.bin_edges[1:] * self.bin_edges[:-1])
        elif bins == "logit":
            self.bin_edges = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
            self.bin_centers = np.array([5e-5, 5e-4, 5e-3, 5e-2, 0.25, 0.75, 1 - 5e-2, 1 - 5e-3, 1 - 5e-4, 1 - 5e-5])
        else:
            self.bin_edges = bins
            self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

        # Perform binning
        freqs = np.arange(1, len(self.counts) + 1) / float(len(self.counts))  # frequency of each count index (i/n_leaves)
        freq_bins = np.digitize(freqs, self.bin_edges, right=True)  # bin that each frequency belongs to
        self.binned = np.bincount(freq_bins, weights=self.counts, minlength=len(self.bin_centers))  # sum mutations in each bin

        if max(self.bin_edges) >= 1.0:
            self.binned = self.binned[1:]  # drop zero bin
        else:
            self.binned = self.binned[1:-1]  # drop zero and one bin

        # self.binned, _ = np.histogram(self.counts, bins=self.bin_edges)  # count mutations per bin

        self.bin_sizes = self.bin_edges[1:] - self.bin_edges[:-1]
        self.binned_normalized = self.binned / self.bin_sizes  # normalize by size of bin

        # Cut SFS at appropriate size for tree by setting all bins beyond 1/leaves to np.nan
        if self.T is not None:
            self.binned_normalized_cut = copy.copy(self.binned_normalized)
            self.binned_normalized_cut[self.bin_edges[1:] < 1/float(len(self.T))] = np.nan
            self.binned_normalized_cut[self.bin_edges[:-1] > 1 - (1/float(len(self.T)))] = np.nan

    def fay_and_wus_H(self):
        """ Get Fay and Wu's H """
        if self._fay_and_wus_H is None:
            self._calculate_fay_and_wus_H()
        return self._fay_and_wus_H

    def _calculate_fay_and_wus_H(self):
        """ Calculate Fay and Wu's H from the mutation counts (site frequency spectrum) """
        n = self.counts.shape[0]  # total number of individuals population
        idx = np.array(range(n))  # number of individuals sharing a mutation
        theta_H = sum(2 * idx**2 * self.counts) / (n * (n - 1))
        theta_pi = sum(2 * idx * (n - idx) * self.counts) / (n * (n - 1))
        self._fay_and_wus_H = theta_pi - theta_H

    def zengs_E(self):
        """ Get Zeng's E """
        if self._zengs_E is None:
            self._calculate_zengs_E()
        return self._zengs_E

    def _calculate_zengs_E(self):
        """ Calculate Zeng's E from the mutation counts (site frequency spectrum) """
        n = self.counts.shape[0]  # total number of individuals population
        idx = np.array(range(n))  # number of individuals sharing a mutation
        a_n = np.log(n) + self.EULERS_CONSTANT + 1/(2*n) - (1/(12*(n**2)))  # (n-1)th harmonic number
        theta_L = sum(idx * self.counts) / (n - 1)
        theta_W = sum(self.counts) / a_n
        self._zengs_E = theta_L - theta_W

    def tajimas_D(self):
        """ Get Tajima's D """
        if self._tajimas_D is None:
            self._calculate_tajimas_D()
        return self._tajimas_D

    def _calculate_tajimas_D(self):
        """ Calculate Tajima's D from the mutation counts (site frequency spectrum) """
        n = self.counts.shape[0]  # total number of individuals population
        idx = np.array(range(n))  # number of individuals sharing a mutation
        a_n = np.log(n) + self.EULERS_CONSTANT + 1/(2*n) - (1/(12*(n**2)))  # (n-1)th harmonic number
        theta_W = sum(self.counts) / a_n
        theta_pi = sum(2 * idx * (n - idx) * self.counts) / (n * (n - 1))
        self._tajimas_D = theta_pi - theta_W

    def ferrettis_L(self):
        """ Get Ferretti's L """
        if self._ferrettis_L is None:
            self._calculate_ferrettis_L()
        return self._ferrettis_L

    def _calculate_ferrettis_L(self):
        """ Calculate Ferretti's L from the mutation counts (site frequency spectrum) """
        n = self.counts.shape[0]  # total number of individuals population
        idx = np.array(range(n))  # number of individuals sharing a mutation
        a_n = np.log(n) + self.EULERS_CONSTANT + 1/(2*n) - (1/(12*(n**2)))  # (n-1)th harmonic number
        theta_W = sum(self.counts) / a_n
        theta_H = sum(2 * idx**2 * self.counts) / (n * (n - 1))
        self._ferrettis_L = theta_W - theta_H
