from jungle.sfs import SFS
# from jungle.forest import Forest
import copy
import itertools
import numpy as np
import pandas as pd
import ete3
from Bio import Phylo
from cStringIO import StringIO
import matplotlib as mpl
import cPickle
import gzip as gz

from .resources.betatree.src import betatree
# TODO move betatree functionality into your own module
# TODO install betatree as package, so that it becomes a dependency, and import as package (# import betatree)

from .resources.FitnessInference.prediction_src import node_ranking
# TODO install FitnessInference as package, so that it becomes a dependency, and import as package (# import FitnessInference)

# TODO move site frequency spectrum and metrics on spectrum to a separate class


class Tree:

    def __init__(self, T, name, params):
        self.T = T
        self.name = name
        self.params = params
        self._site_frequency_spectrum = None
        self._fay_and_wus_H = None
        self._zengs_E = None
        self._tajimas_D = None
        self._ferrettis_L = None
        self._num_descendants = None
        self._variance_descendants = None
        self._gini_descendants = None
        self.T_fitness = None
        self.is_depth_annotated = False
        self.colless = None

    def __len__(self):
        return len(self.T)

    @classmethod
    def from_newick(cls, filename, name=None, params=None):
        """ Construct Tree from newick file """
        T = ete3.Tree(filename, format=1)
        T.ladderize()
        Tree.name_internal_nodes(T)
        return Tree(T, name, params)

    @classmethod
    def from_pickle(cls, filename, gzip=None):
        """ Load Tree from pickle file """
        if gzip is None and (".gz" in filename or ".gzip" in filename):
            gzip = True
        if gzip:
            with gz.open(filename, 'rb') as f:
                return cPickle.load(f)
        else:
            with open(filename, 'rb') as f:
                return cPickle.load(f)

    @classmethod
    def generate(cls, params, name=None):
        """ Generate tree by simulation """
        # TODO write your own code to generate tree
        T_betatree = betatree.betatree(sample_size=params["n_leaves"], alpha=params["alpha"])
        T_betatree.coalesce()
        T = ete3.Tree(T_betatree.BioTree.format(format="newick"), format=1)
        T.ladderize()
        Tree.name_internal_nodes(T)
        return Tree(T, name, params)

    def to_newick(self, outfile, **kwargs):
        """ Write Tree to Newick """
        # format=3 includes all branches with names (but no supports)
        # To export features, use attribute features=[feature1, feature2]
        self.T.write(outfile=outfile, format=3, **kwargs)

    def annotate_standard_node_features(self):
        """ Annotate each node of Tree with standard features:
            depth, depth_rank, depth_normalized, num_children, num_descendants
        """

        for depth_rank, node in enumerate(self.T.traverse("postorder")):

            depth = node.get_distance(self.T)  # depth is distance to root
            setattr(node, "depth", depth)
            node.features.add("depth")

            # depth rank has been removed because T.traverse("postorder") is much more efficient
            # and rank now must be calculated separately
            # setattr(node, "depth_rank", depth_rank)
            # node.features.add("depth_rank")

            num_children = len(node.get_children())
            setattr(node, "num_children", num_children)
            node.features.add("num_children")

            num_descendants = len(node.get_children()) + sum(child.num_descendants for child in node.get_children())
            setattr(node, "num_descendants", num_descendants)
            node.features.add("num_descendants")

            num_leaf_descendants = len(node.get_leaves())
            setattr(node, "num_leaf_descendants", num_leaf_descendants)
            node.features.add("num_leaf_descendants")

        self.is_depth_annotated = True  # set flag to indicate that depth has been calculated

        max_depth = self.max_depth()

        # calculate normalized depth
        for _, node in enumerate(self.T.traverse("levelorder")):

            depth_normalized = node.depth / float(max_depth)
            setattr(node, "depth_normalized", depth_normalized)
            node.features.add("depth_normalized")

    def node_features(self, subset=None):
        """ Get DataFrame of features for each node """

        # Determine features to get
        if subset is None:
            # get all features that exist for any node
            features = set().union(*[node.features for node in self.T.traverse("levelorder")])
        else:
            features = subset

        # Get features and write into DataFrame

        result = pd.DataFrame()  # DataFrame of features

        for attr in features:

            values = []

            for node in self.T.traverse("levelorder"):
                if hasattr(node, attr):
                    value = getattr(node, attr)
                else:
                    value = None
                values.append(value)

            result[attr] = values

        # Add feature that specifies whether each node is a leaf
        result["is_leaf"] = [node.is_leaf() for node in self.T.traverse("levelorder")]

        return result

    def annotate_imbalance(self):
        """ Annotate each node of Tree with its imbalance I.
            If N is a list of the number of descendants of each child of the node, then
            I = max(N) / sum(N).
        """

        for _, node in enumerate(self.T.traverse("levelorder")):

            if node.is_leaf():
                # imbalance of a leaf is not defined
                imbalance = np.nan
            else:
                # get number of descendants of each child N
                num_descendants_of_children = [len(x) for x in node.get_children()]
                # calculate imbalance I = max(N) / sum(N)
                imbalance = max(num_descendants_of_children) / float(sum(num_descendants_of_children))

            setattr(node, "imbalance", imbalance)
            node.features.add("imbalance")

    def annotate_colless(self):
        """ Annotate each node of Tree with its Colless index. """

        for _, node in enumerate(self.T.traverse("postorder")):

            if node.is_leaf():

                # colless of a leaf is not defined
                difference_num_descendants = np.nan                
                colless = 0

            else:

                children = node.get_children()

                if len(children) != 2:
                    print "Error: Tree must be binary to calculate Colless index. Use Tree.resolve_polytomy() to binarize."
                    return None

                child1 = children[0]
                child2 = children[1]
                difference_num_descendants = np.abs(child1.num_descendants - child2.num_descendants)
                colless = difference_num_descendants + child1.colless + child2.colless

            log10p_colless = np.log10(colless + 1)

            setattr(node, "difference_num_descendants", difference_num_descendants)
            node.features.add("difference_num_descendants")                
                
            setattr(node, "colless", colless)
            node.features.add("colless")

            setattr(node, "log10p_colless", log10p_colless)
            node.features.add("log10p_colless")

        # Set attribute to be accessible from top-level Tree object
        self.colless = self.T.colless
        self.log10p_colless = self.T.log10p_colless
            
    def max_depth(self):
        """ Get maximum depth of the tree """
        if self.is_depth_annotated is False:
            print("Error: Tree.annotate_standard_node_features() must be run before max_depth().")
            return None
        else:
            depths = [node.depth for node in self.T.traverse()]
            return max(depths)

    def total_branch_length(self):
        """ Get total branch length of tree """
        return np.sum([node.dist for node in self.T.traverse()])

    def rescale(self, total_branch_length):
        """ Rescales tree so that total branch length equals given number """
        total_branch_length_initial = self.total_branch_length()
        scaling_factor = total_branch_length / total_branch_length_initial
        for node in self.T.traverse():
            node.dist = node.dist * scaling_factor

    def resolve_polytomy(self):
        """ Resolve all polytomies in Tree by creating an arbitrary dicotomic structure """
        self.T.resolve_polytomy()
        self.T.ladderize()
            
    def site_frequency_spectrum(self):
        """ Get site frequency spectrum (SFS) of Tree """
        if self._site_frequency_spectrum is None:
            self._site_frequency_spectrum = SFS.from_tree(self.T)
        return self._site_frequency_spectrum

    def bin_site_frequency_spectrum(self, bins, *args, **kwargs):
        """ Bin the site frequency spectrum (SFS) of Tree """
        self._site_frequency_spectrum.bin(bins, *args, **kwargs)
        return self._site_frequency_spectrum.binned_normalized_cut

    def fay_and_wus_H(self):
        """ Get Fay and Wu's H """
        if self._fay_and_wus_H is None:
            self._fay_and_wus_H = self._site_frequency_spectrum.fay_and_wus_H()
        return self._fay_and_wus_H

    def zengs_E(self):
        """ Get Zeng's E """
        if self._zengs_E is None:
            self._zengs_E = self._site_frequency_spectrum.zengs_E()
        return self._zengs_E

    def tajimas_D(self):
        """ Get Tajima's D """
        if self._tajimas_D is None:
            self._tajimas_D = self._site_frequency_spectrum.tajimas_D()
        return self._tajimas_D

    def ferrettis_L(self):
        """ Get Ferretti's L """
        if self._ferrettis_L is None:
            self._ferrettis_L = self._site_frequency_spectrum.ferrettis_L()
        return self._ferrettis_L

    def infer_fitness(self, params={}):
        """ Infer fitness metrics of each node.
            Uses FitnessInference package by Richard Neher (eLife 2014).
            The FitnessInference package uses Biopython trees, so we start
            by constructing a Biopython tree version of our tree. Then, we
            perform the fitness inference, which calculates several fitness
            metrics of each node. Then we copy the results back into our
            ete3 tree object.
        """

        # Construct a Biopython tree
        T_biopython = Phylo.read(StringIO(self.T.write(format=1)), "newick")  # create Biopython tree
        Tree.repair_branch_lengths(T_biopython)  # set missing branch lengths to zero (needed because some branch lengths can occasionally be missing)

        # Determine depth of the tree (used to set time bins for the expansion score metric)
        _, depth = self.T.get_farthest_leaf()
        time_bins = np.linspace(0, depth, 20)

        # Perform fitness inference
        self.T_fitness = node_ranking.node_ranking(methods=['mean_fitness', 'polarizer', 'expansion_score'], time_bins=time_bins)
        self.T_fitness.set_tree(T_biopython)
        self.T_fitness.compute_rankings()
        self.T_fitness.rank_by_method(nodes=T_biopython.get_nonterminals(), method='mean_fitness')
        self.T_fitness.rank_by_method(nodes=T_biopython.get_terminals(), method='mean_fitness')

        # Copy results into ete3 tree
        for node in self.T_fitness.T.get_nonterminals() + self.T_fitness.T.get_terminals():

            # Find corresponding node in ete3 tree
            nodes_found = self.T.search_nodes(name=node.name)
            if len(nodes_found) > 0:
                node_ete3 = self.T.search_nodes(name=node.name)[0]
            else:
                continue

            # Copy each attribute
            for attr in ['mean_fitness', 'var_fitness', 'expansion_score', 'rank']:
                if hasattr(node, attr):
                    setattr(node_ete3, attr, getattr(node, attr))  # set attribute of node
                    node_ete3.features.add(attr)  # add attribute to list of features for node

        # Delete fitness object
        # Unfortunately, the node_ranking class raises errors when pickled and cannot be pickled
        del self.T_fitness

    def variance_descendants(self):
        """ Get variance of number of descendants (variance of number of descendants across branches).
        Proposed by Ferretti et al. 2017 Genetics.
        """
        if self._variance_descendants is None:
            self._variance_descendants = self._calculate_variance_descendants()
        return self._variance_descendants

    def _calculate_variance_descendants(self):
        """" Calculate variance of number of descendants """
        if self._num_descendants is None:
            self._num_descendants = np.array([len(node) for node in self.T.traverse("levelorder")])
        return np.var(self._num_descendants)

    def gini_descendants(self):
        """ Get Gini coefficient of number of descendants. """
        if self._gini_descendants is None:
            self._gini_descendants = self._calculate_gini_descendants()
        return self._gini_descendants

    def _calculate_gini_descendants(self):
        """ Calculate Gini coefficient of number of descendants. """
        if self._num_descendants is None:
            self._num_descendants = np.array([len(node) for node in self.T.traverse("levelorder")])
        return _gini(self._num_descendants.astype(np.float16))

    def gini_by_depth(self):
        pass

    def selection_on_subclones(self):
        pass

    def generate_history(self):
        """ Generate Forest composed of Trees which replay the history of this Tree """

        trees_forward = []  # list of trees which will become the Forest
        T_forward = ete3.Tree()  # initialize as empty tree

        for node_original in itertools.islice(self.T.traverse("levelorder"), 1, None):

            node = copy.deepcopy(node_original)  # copy node (so we do not change original object)

            # remove children of node
            for child in node.get_children():
                node.remove_child(child)

            # add node to growing tree
            search_results = T_forward.iter_search_nodes(name=node.up.name)  # find ancestor
            next(search_results).add_child(node)  # add node as child of ancestor

            T_forward_frozen = copy.deepcopy(T_forward)
            T_forward_frozen_as_jungle_tree = Tree(T_forward_frozen, name=None, params=None)
            trees_forward.append(T_forward_frozen_as_jungle_tree)

        return trees_forward

    def pvalue(self, attribute, model, strict_bounds=True, invert_cdf=False, suffix=None):
        """ Calculate P value of attribute under model for each node.
            Stores result in [attribute]_pvalue_[suffix] or
            [attribute]_pvalue_[model.name] if suffix is None.
        """

        if suffix is None:
            suffix = model.name

        if model.name is None and suffix is None:
            suffix = ""
            print("Warning: suffix and model.name are None, so the pvalue columns will be labeled with empty suffix")

        label_pvalue = attribute + "_pvalue_" + suffix
        label_model_mean = attribute + "_model_mean_" + suffix        

        for node in self.T.traverse("levelorder"):

            x = getattr(node, attribute)
            size = getattr(node, "num_leaf_descendants")
            pvalue = model.pvalue(x=x, size=size, strict_bounds=strict_bounds, invert_cdf=invert_cdf)
            model_mean = model.model_mean(size=size, strict_bounds=strict_bounds)

            setattr(node, label_pvalue, pvalue)
            node.features.add(label_pvalue)

            setattr(node, label_model_mean, model_mean)
            node.features.add(label_model_mean)            

        return None
    
    def color(self, by=None, cmap=mpl.cm.coolwarm, norm=mpl.colors.Normalize(), fill_leaves=True):
        """ Color nodes in tree by an attribute """

        # Get attribute of each node

        attr_dict = {}
        color_dict = {}

        for node in self.T.traverse():
            if hasattr(node, by):
                val = getattr(node, by) # value of attribute
                attr_dict[node] = val
            else:
                # Set color of missing values
                color = "#000000"
                color_dict[node] = color

        # Normalize attribute values and get colors
        vals_normalized = norm(attr_dict.values()) # compute normalized values
        colors_rgb = cmap(vals_normalized) # map normalized values to colors
        colors = [mpl.colors.to_hex(c) for c in colors_rgb] # convert to hex
        color_dict.update(dict(zip(attr_dict.keys(), colors))) # add colors to dictionary that already has missing values

        # Fill colors of each leaf by its parent
        if fill_leaves:
            for node in self.T.traverse():
                if node.is_leaf():
                    color_dict[node] = color_dict[node.up]

        # Set style of each node

        for node in self.T.traverse():

            nstyle = ete3.NodeStyle()

            # Turn off symbol
            # nstyle["fgcolor"] = "#000000"
            nstyle["size"] = 0

            # Set solid lines
            nstyle["hz_line_type"] = 0

            # Set line width
            nstyle["hz_line_width"] = 2
            nstyle["vt_line_width"] = 2

            # Set color by attribute
            color = color_dict[node]
            nstyle["hz_line_color"] = color
            nstyle["vt_line_color"] = color

            node.set_style(nstyle)

    def render(self, *args, **kwargs):
        """ Render tree as image """
        return self.T.render(*args, **kwargs)

    @staticmethod
    def repair_branch_lengths(T_biopython):
        """ Sets missing branch lengths of a Biopython tree to zero """
        for clade in T_biopython.get_terminals():
            if clade.branch_length is None:
                clade.branch_length = 0.0
        for clade in T_biopython.get_nonterminals():
            if clade.branch_length is None:
                clade.branch_length = 0.0

    @staticmethod
    def name_internal_nodes(T):
        """ Assign names to internal nodes """
        for i, node in enumerate(T.iter_descendants("levelorder")):
            if not node.is_leaf():
                node.name = "_" + str(i)

    def dump(self, filename, gzip=None):
        """ Save Tree to disk
            If filename contains ".gz" or ".gzip", it will be gzipped.
        """
        if gzip is None and (".gz" in filename or ".gzip" in filename):
            gzip = True
        if gzip:
            with gz.open(filename, 'wb') as f:
                cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            with open(filename, 'wb') as f:
                cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

def _gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient
