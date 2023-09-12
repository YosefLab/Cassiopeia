import gzip as gz
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
import scipy
from jungle.tree import Tree


class Forest:
    def __init__(self, trees=[], name=None, params=None):
        self.trees = trees
        self.name = name
        self.params = params
        self._site_frequency_spectrum = None
        self._fay_and_wus_H = None
        self._zengs_E = None
        self._tajimas_D = None
        self._ferrettis_L = None

    def __len__(self):
        return len(self.trees)

    @classmethod
    def from_newick(cls, filenames, name=None, params=None):
        """Construct Forest by loading Trees from newick file"""
        trees = []
        for filename in filenames:
            t = Tree.from_newick(filename)
            trees.append(t)
        return Forest(trees, name, params)

    @classmethod
    def from_pickle(cls, filename, gzip=None):
        """Load Forest from pickle file"""
        if gzip is None and (".gz" in filename or ".gzip" in filename):
            gzip = True
        if gzip:
            with gz.open(filename, "rb") as f:
                return pickle.load(f)
        else:
            with open(filename, "rb") as f:
                return pickle.load(f)

    @classmethod
    def generate(cls, n_trees, name=None, params=None):
        """Generate Forest by simulating Trees"""
        trees = []
        for _ in range(n_trees):
            t = Tree.generate(params=params)
            trees.append(t)
        return Forest(trees, name, params)

    @classmethod
    def from_newick_tar_gz(cls, filename, name=None, params=None):
        """Load Forest from a gzipped tar of Newick files"""
        trees = []
        with tarfile.open(filename, "r:gz") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                content = f.read()
                if f is not None:
                    t = Tree.from_newick(content)
                    trees.append(t)
        return Forest(trees, name, params)

    def to_newick(self, outfile, **kwargs):
        """Write Tree to a gzipped tar of Newick files"""
        # format=3 includes all branches with names (but no supports)
        # To export features, use attribute features=[feature1, feature2]

        # Write Trees to Newick files
        outfile_trees = ["tree" + str(i) for i in range(len(self.trees))]
        for tree, outfile_tree in zip(self.trees, outfile_trees):
            tree.to_newick(outfile=outfile_tree, **kwargs)

        # Make tar.gz out of Newick files
        with tarfile.open(outfile, "w:gz") as tar:
            for outfile_tree in outfile_trees:
                tar.add(outfile_tree)

        # Delete Newick files
        for outfile_tree in outfile_trees:
            if os.path.exists(outfile_tree):
                os.remove(outfile_tree)

    def concat(self, other):
        """Concatenate two Forests. Concatenates lists of trees and attributes."""

        trees = self.trees + other.trees
        name = None
        params = None

        F_new = Forest(trees, name, params)

        # Concatenate all list-like attributes
        attrs = [
            "_site_frequency_spectrum",
            "_fay_and_wus_H",
            "_zengs_E",
            "_tajimas_D",
            "_ferrettis_L",
        ]
        for attr in attrs:
            attr_value = _concat_lists_safe(
                getattr(self, attr), getattr(other, attr)
            )
            setattr(
                F_new, attr, attr_value
            )  # set new attribute to fit parameters

        return F_new

    def annotate_standard_node_features(self):
        """Annotate each node of each Tree in Forest with standard features:
        depth, depth_rank, num_children, num_descendants
        """
        for tree in self.trees:
            tree.annotate_standard_node_features()

    def annotate_imbalance(self):
        """Annotate each node of each Tree in Forest with its imbalance I.
        If N is a list of the number of descendants of each child of the node, then
        I = max(N) / sum(N).
        """
        for tree in self.trees:
            tree.annotate_imbalance()

    def annotate_colless(self):
        """Annotate each node of each Tree in Forest with its Colless index."""
        for tree in self.trees:
            tree.annotate_colless()

    def node_features(self, subset=None):
        """Get DataFrame of features for each node of each Tree in Forest"""
        features = [tree.node_features(subset) for tree in self.trees]
        keys = [tree.name for tree in self.trees]
        names = ["name_tree", "id_node"]
        # ensure tree names are unique
        if len(keys) != len(set(keys)):
            # tree names are not unique, so use unique numbering as id
            keys_unique = list(range(len(keys)))
        else:
            keys_unique = keys
        result = pd.concat(
            features, keys=keys_unique, names=names, ignore_index=False
        )
        return result

    def rescale(self, total_branch_length):
        """Rescales all Trees in Forest so that total branch length equals given number"""
        for tree in self.trees:
            tree.rescale(total_branch_length=total_branch_length)

    def resolve_polytomy(self):
        """Resolve all polytomies by creating an arbitrary dicotomic structure in all Trees in Forest"""
        for tree in self.trees:
            tree.resolve_polytomy()

    def site_frequency_spectrum(self):
        """Get site frequency spectrum (SFS) of all Trees in Forest"""
        self._site_frequency_spectrum = []
        for tree in self.trees:
            self._site_frequency_spectrum.append(tree.site_frequency_spectrum())
        return self._site_frequency_spectrum

    def bin_site_frequency_spectrum(self, bins, *args, **kwargs):
        """Bin the site frequency spectrum (SFS) of all Trees in Forest"""
        S = []
        for tree in self.trees:
            s = tree.bin_site_frequency_spectrum(bins, *args, **kwargs)
            S.append(s)
        return S

    def mean_site_frequency_spectrum(self, which="binned_normalized_cut"):
        """Get mean site frequency spectrum (SFS) of all Trees in Forest"""
        S = np.empty(
            (
                len(self.trees),
                getattr(self._site_frequency_spectrum[0], which).shape[0],
            )
        )  # array of binned SFS (dimensions are (number of trees, number of bins))
        for i, s in enumerate(self._site_frequency_spectrum):
            S[i, :] = getattr(s, which)
        mean = np.nanmean(S, axis=0)
        sem = scipy.stats.sem(S, axis=0)
        return mean, sem

    def fay_and_wus_H(self):
        """Get Fay and Wu's H of all Trees in Forest"""
        self._fay_and_wus_H = []
        for tree in self.trees:
            self._fay_and_wus_H.append(tree.fay_and_wus_H())
        return self._fay_and_wus_H

    def zengs_E(self):
        """Get Zeng's E of all Trees in Forest"""
        self._zengs_E = []
        for tree in self.trees:
            self._zengs_E.append(tree.zengs_E())
        return self._zengs_E

    def tajimas_D(self):
        """Get Tajima's D of all Trees in Forest"""
        self._tajimas_D = []
        for tree in self.trees:
            self._tajimas_D.append(tree.tajimas_D())
        return self._tajimas_D

    def ferrettis_L(self):
        """Get Ferretti's L of all Trees in Forest"""
        self._ferrettis_L = []
        for tree in self.trees:
            self._ferrettis_L.append(tree.ferrettis_L())
        return self._ferrettis_L

    def fit_metric(self, attr, model):
        """Fit model to distribution of metric"""
        if not hasattr(self, attr) or getattr(self, attr) == None:
            print(
                ("Error in Forest.fit_metric(): Forest has no attribute", attr)
            )
            return None
        metric = getattr(self, attr)  # get values of metric
        fit = model.fit(metric)  # fit model to values of metric
        attr_name = attr + "_fit"
        setattr(self, attr_name, fit)  # set new attribute to fit parameters
        return getattr(self, attr_name)

    def p_metric(self, attr, model, value):
        fit = self.fit_metric(attr, model)  # fit model
        p = model.cdf(
            value, *fit
        )  # calculate pvalue of value under model with fitted parameters
        return p

    def infer_fitness(self, *args, **kwargs):
        """Infer fitness metrics of each node of all Trees in Forest"""
        for tree in self.trees:
            tree.infer_fitness(*args, **kwargs)

    def variance_descendants(self):
        """Get variance of number of descendants of all Trees in Forest"""
        self._variance_descendants = []
        for tree in self.trees:
            self._variance_descendants.append(tree.variance_descendants())
        return self._variance_descendants

    def gini_descendants(self):
        """Get Gini coefficient of number of descendants of all Trees in Forest"""
        self._gini_descendants = []
        for tree in self.trees:
            self._gini_descendants.append(tree.gini_descendants())
        return self._gini_descendants

    def gini(self):
        pass

    def gini_by_depth(self):
        pass

    def selection_on_subclones(self):
        pass

    def pvalue(
        self,
        attribute,
        model,
        strict_bounds=True,
        invert_cdf=False,
        suffix=None,
    ):
        for tree in self.trees:
            tree.pvalue(attribute, model, strict_bounds, invert_cdf, suffix)

    def color(self, *args, **kwargs):
        """Color nodes in all Trees of Forest by an attribute"""
        for tree in self.trees:
            tree.color(*args, **kwargs)

    def render(self, filenames, *args, **kwargs):
        """Render all Trees of Forest as images"""
        for tree, filename in zip(self.trees, filenames):
            tree.render(filename, *args, **kwargs)

    def dump(self, filename, gzip=None):
        """Save Forest to disk
        If filename contains ".gz" or ".gzip", it will be gzipped.
        """
        if gzip is None and (".gz" in filename or ".gzip" in filename):
            gzip = True
        if gzip:
            with gz.open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


def unique_list_str(L):
    """Ensure each element of a list of strings is unique by appending a number to duplicates.
    Note that this fails to generate uniqueness if a trio "Name", "Name", "Name_1" exists.
    """
    L_unique = []
    count = {}
    for s in L:
        if s in count:
            s_unique = str(s) + "_" + str(count[s])
            count[s] += 1
        else:
            s_unique = str(s)
            count[s] = 1
        L_unique.append(s_unique)
    return L_unique


def _concat_lists_safe(A, B):
    """Safely concatenate two lists if one or both may be None"""

    if A is None and B is None:
        return None
    elif isinstance(A, list) and B is None:
        return A
    elif isinstance(B, list) and A is None:
        return B
    elif isinstance(A, list) and isinstance(B, list):
        return A + B
    else:
        raise TypeError("A and B must be lists or None")
