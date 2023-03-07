import os
import sys
import tempfile
from typing import Optional

import networkx as nx

dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"dir_path = {dir_path}")
sys.path.append(os.path.join(dir_path, "_jungle"))
import jungle as jg
import numpy as np

from cassiopeia.data import CassiopeiaTree

from ._FitnessEstimator import FitnessEstimator, FitnessEstimatorError


def _to_newick(tree: nx.DiGraph, record_branch_lengths: bool = False) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree
        record_branch_lengths: Whether to record branch lengths on the tree in
            the newick string

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        weight_string = ""

        if record_branch_lengths and g.in_degree(node) > 0:
            parent = list(g.predecessors(node))[0]
            weight_string = ":" + str(g[parent][node]["length"])
            if not is_leaf:
                weight_string = node + weight_string

        _name = str(node)
        return (
            "%s" % (_name,) + weight_string
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
                + weight_string
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


class LBIJungle(FitnessEstimator):
    """
    LBI as implemented by the jungle package.

    Implements the LBI fitness estimator described by Neher et al. (2014).
    This is a simple wrapper on top of the Jungle package, which is in turn
    a wrapper around Neher et al.'s code.

    Caveat: LBIJungle does not estimate fitness for the root of this tree
    (artifact of the Jungle package). This is rarely of interest though.

    Args:
        random_seed: Random seed to set in numpy before running fitness
            estimates. (A random seed is used by the LBI to estimate the
            characteristic timescale `tau` of the underlying process.
            See Neher et al. 2014, and the LBIJungle package for details.)
    """

    def __init__(self, random_seed: Optional[int] = None):
        self._random_seed = random_seed

    def estimate_fitness(self, tree: CassiopeiaTree) -> None:
        """
        Sets attribute `fitness` for each node in the tree using the LBI.

        Caveat: LBIJungle does not estimate fitness for the root of this tree
        (artifact of the Jungle package). This is rarely of interest though.

        Will raise a FitnessEstimatorError if the CassiopeiaTree cannot be
        serialized to networkx.

        Raises:
            FitnessEstimatorError
        """
        with tempfile.NamedTemporaryFile("w") as outfile:
            outfilename = outfile.name
            tree_newick = _to_newick(
                tree.get_tree_topology(), record_branch_lengths=True
            )
            outfile.write(tree_newick)
            outfile.flush()
            if self._random_seed is not None:
                np.random.seed(self._random_seed)
            try:
                T_empirical = jg.Tree.from_newick(outfilename)
            except Exception:
                raise Exception(f"Could not read newick str:\n{tree_newick}")
            T_empirical.annotate_standard_node_features()
            T_empirical.infer_fitness(params={})
            res_df = T_empirical.node_features()
            node_names = res_df.name
            node_fitnesses = res_df.mean_fitness
            for v, f in zip(node_names, node_fitnesses):
                if v != "" and v[0] != "_":
                    tree.set_attribute(v, "fitness", f)
                elif v != "" and v[0] == "_":
                    # (Non-root) internal node!
                    tree.set_attribute(v[1:], "fitness", f)
