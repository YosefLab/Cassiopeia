"""
This file stores a subclass of BranchLengthEstimator, the
IIDExponentialBayesian. Briefly, this model assumes that CRISPR/Cas9 mutates
each site independently and identically, with an exponential waiting time;
and that the phylogeny follows a subsampled Birth Process. Conditional on the
observed topology and on the character data, the posterior mean branch lengths
are used as branch length estimates.
"""
from copy import deepcopy
from typing import Dict, List

import numpy as np

from cassiopeia.data import CassiopeiaTree

from ._iid_exponential_bayesian import _PyInferPosteriorTimes
from .BranchLengthEstimator import BranchLengthEstimator


def _get_number_of_mutated_characters_in_node(tree: CassiopeiaTree, v: str):
    """Mutated characters in node v, excluding missing characters."""
    states = tree.get_character_states(v)
    return len(
        [s for s in states if s != 0 and s != tree.missing_state_indicator]
    )


def _non_root_internal_nodes(tree: CassiopeiaTree) -> List[str]:
    """Internal nodes of the tree, excluding the root."""
    return list(set(tree.internal_nodes) - set(tree.root))


class IIDExponentialBayesian(BranchLengthEstimator):
    """
    Inference under a subsampled Birth Process with IID memoryless mutations.

    In more detail, his model assumes that CRISPR/Cas9 mutates each site
    independently and identically, with an exponential waiting time; and that
    the phylogeny follows a subsampled Birth Process. It further assumes that
    the tree has depth exactly 1. Conditional on the observed topology and on
    the character data, the posterior mean branch lengths are used as branch
    length estimates.

    This estimator requires that the ancestral characters be provided (these
    can be imputed with CassiopeiaTree's reconstruct_ancestral_characters
    method if they are not known, which is usually the case for real data).

    This estimator also assumes that the tree is binary (except for the root,
    which should have degree 1).

    Missing states are treated as missing at random by the model.

    Args:
        mutation_rate: The CRISPR/Cas9 mutation rate.
        birth_rate: The phylogeny birth rate.
        sampling_probability: The probability that a leaf in the ground truth
            tree was sampled. Must be in (0, 1]
        discretization_level: How many timesteps are used to discretize time.

    Attributes:
        mutation_rate: The CRISPR/Cas9 mutation rate.
        birth_rate: The phylogeny birth rate.
        sampling_probability: The probability that a leaf in the ground truth
            tree was sampled.
        discretization_level: How many timesteps are used to discretize time.
        log_likelihood: The log-likelihood of the training data under the
            model.
    """

    def __init__(
        self,
        mutation_rate: float,
        birth_rate: float,
        sampling_probability: float,
        discretization_level: int = 600,
    ):
        if sampling_probability <= 0 or sampling_probability > 1:
            raise ValueError(
                "sampling_probability should be in (0, 1]: "
                f"{sampling_probability} provided."
            )
        self._mutation_rate = mutation_rate
        self._birth_rate = birth_rate
        self._sampling_probability = sampling_probability
        self._discretization_level = discretization_level
        self._log_likelihood = None

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """
        Estimate branch lengths of the tree using the given model.

        The tree must be binary except for the root, which should have degree
        1.

        This method raises a ValueError if the discretization_size is too
        small or the tree topology is not valid.

        The computational complexity of this method is:
        O(discretization_level * tree.n_cell * tree.n_character)

        Raises:
            ValueError if discretization_size is too small or the tree topology
            is not valid.
        """
        self._validate_input_tree(tree)
        self._tree_orig = tree

        tree = deepcopy(tree)
        # We first impute the unambiguous missing states because it makes
        # the number of mutated states at each vertex increase monotonically
        # from parent to child, making the dynamic programming state and code
        # much clearer.
        tree.impute_deducible_missing_states()

        self._precompute_K_non_missing(tree)
        self._log_joints = {}  # type: Dict[str, np.array]
        self._posterior_means = {}  # type: Dict[str, float]
        self._posteriors = {}  # type: Dict[str, np.array]
        self._tree = tree

        self._populate_attributes_with_cpp_implementation()
        self._populate_branch_lengths()

    def _populate_branch_lengths(self):
        """
        Populate the branch lengths of the tree using the posterior means.
        """
        tree = self._tree_orig
        times = {}
        for node in _non_root_internal_nodes(tree):
            times[node] = self._posterior_means[node]
        times[tree.root] = 0.0
        for leaf in tree.leaves:
            times[leaf] = 1.0
        tree.set_times(times)

    @staticmethod
    def _validate_input_tree(tree):
        """
        Checks that the tree is binary, except for the root node, which should
        have degree exactly 1.

        Raises:
            ValueError
        """
        if len(tree.children(tree.root)) != 1:
            raise ValueError(
                "The root of the tree should have degree exactly " "1."
            )
        for node in tree.internal_nodes:
            if node != tree.root and len(tree.children(node)) != 2:
                raise ValueError(
                    "Each internal node (different from the root)"
                    " should have degree exactly 2."
                )

    def _populate_attributes_with_cpp_implementation(self):
        """
        Run c++ implementation that infers posterior node times.

        Wrapper that calls a fast c++ implementation to compute the
        _log_joints, _posterior_means, _posteriors, _log_likelihood.
        The key here is that the nodes of the tree are mapped to integer ids
        from 0 to tree.n_cell - 1. This mapping is then undone at the end of
        this method.
        """
        # First extract the relevant information from the tree to pass on to
        # the c++ module.
        tree = self._tree

        node_to_id = dict(zip(tree.nodes, range(len(tree.nodes))))
        id_to_node = dict(zip(range(len(tree.nodes)), tree.nodes))

        N = len(tree.nodes)

        # children[i] holds the children of node i in the tree.
        children = sorted(
            [
                [node_to_id[v]] + [node_to_id[c] for c in tree.children(v)]
                for v in tree.nodes
            ]
        )
        children = [vec[1:] for vec in children]

        # The id of the root of the tree.
        root = node_to_id[tree.root]

        # is_internal_node[i] is a binary indicator telling whether node i
        # is internal in the tree.
        is_internal_node = np.array(
            sorted(
                [
                    [node_to_id[v], 1 * tree.is_internal_node(v)]
                    for v in tree.nodes
                ]
            )
        )[:, 1]

        # get_n_mut_chars_in_node[i] is the number of
        # mutated characters (this excludes missing characters) in node i.
        get_n_mut_chars_in_node = np.array(
            sorted(
                [
                    [
                        node_to_id[v],
                        _get_number_of_mutated_characters_in_node(tree, v),
                    ]
                    for v in tree.nodes
                ]
            )
        )[:, 1]

        # The list of internal nodes different from the root.
        non_root_internal_nodes = [
            node_to_id[v] for v in _non_root_internal_nodes(tree)
        ]

        # The leaves of the tree.
        leaves = [node_to_id[v] for v in tree.leaves]

        # parent[i] contains the parent of node i.
        parent = np.array(
            sorted(
                [
                    [node_to_id[v], node_to_id[tree.parent(v)]]
                    for v in tree.nodes
                    if v != tree.root
                ]
                + [[node_to_id[tree.root], -100000000]]
            )
        )[:, 1]

        # The number of characters.
        K = tree.n_character

        # The number of characters, after ignoring missing characters.
        K_non_missing = np.array(
            sorted(
                [[node_to_id[v], self._K_non_missing[v]] for v in tree.nodes]
            )
        )[:, 1]

        # The number of timesteps used to discretize time.
        T = self._discretization_level

        # The mutation rate.
        r = self._mutation_rate

        # The birth rate
        lam = self._birth_rate

        # The probability of each leaf being sampled.
        sampling_probability = self._sampling_probability

        # is_leaf[i] is a binary indicator for whether i is a leaf.
        is_leaf = np.array(
            sorted([[node_to_id[v], 1 * tree.is_leaf(v)] for v in tree.nodes])
        )[:, 1]

        # Now we pass in all the tree data to the c++ implementation.
        infer_posterior_times = _PyInferPosteriorTimes()
        infer_posterior_times.run(
            N=N,
            children=children,
            root=root,
            is_internal_node=is_internal_node,
            get_number_of_mutated_characters_in_node=get_n_mut_chars_in_node,
            non_root_internal_nodes=non_root_internal_nodes,
            leaves=leaves,
            parent=parent,
            K=K,
            K_non_missing=K_non_missing,
            T=T,
            r=r,
            lam=lam,
            sampling_probability=sampling_probability,
            is_leaf=is_leaf,
        )

        self._log_likelihood = infer_posterior_times.get_log_likelihood_res()

        for key_value in infer_posterior_times.get_log_joints_res():
            if len(key_value) != 2:
                raise ValueError("get_log_joints_res should return pairs")
            key = key_value[0]
            value = key_value[1]
            if len(value) != self._discretization_level + 1:
                raise ValueError("log-joints should have length T + 1")
            self._log_joints[id_to_node[key]] = np.array(value)

        for key_value in infer_posterior_times.get_posteriors_res():
            if len(key_value) != 2:
                raise ValueError("get_posteriors_res should return pairs")
            key = key_value[0]
            value = key_value[1]
            if len(value) != self._discretization_level + 1:
                raise ValueError("posteriors should have length T + 1")
            self._posteriors[id_to_node[key]] = np.array(value)

        for key_value in infer_posterior_times.get_posterior_means_res():
            if len(key_value) != 2:
                raise ValueError("posterior means should return pairs")
            key = key_value[0]
            value = key_value[1]
            self._posterior_means[id_to_node[key]] = np.array(value)

    def _precompute_K_non_missing(self, tree: CassiopeiaTree):
        """
        For each vertex in the tree, how many states are not missing.
        """
        # Check precondition: Add deducible states must have been imputed.
        for (parent, child) in tree.edges:
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            for (parent_state, child_state) in zip(parent_states, child_states):
                # Check that deducible missing states have been imputed.
                # (This should ALWAYS pass)
                if (
                    parent_state != 0
                    and parent_state != tree.missing_state_indicator
                    and child_state == tree.missing_state_indicator
                ):
                    raise ValueError(
                        "Some deducible missing states have not "
                        "been imputed."
                    )

        # Compute _K_non_missing
        self._K_non_missing = {}
        self._K_non_missing[tree.root] = tree.n_character
        for node in tree.nodes:
            self._K_non_missing[
                node
            ] = tree.n_character - tree.get_character_states(node).count(
                tree.missing_state_indicator
            )

        # Validate monotonicity of K_non_missing
        for (parent, child) in tree.edges:
            if self._K_non_missing[parent] < self._K_non_missing[child]:
                raise ValueError(
                    "The number of missing states is not " "monotone."
                )

    @property
    def log_likelihood(self):
        """
        The log-likelihood of the training data under the estimated model.
        """
        return self._log_likelihood

    def log_joints(self, node: str) -> np.array:
        """
        log joint node time probabilities.

        The log joint probability density of the observed tree topology, state
        vectors, and all possible times for the node. In other words:
        log P(node time = t, character states, tree topology) for t in [0, T]
        where T is the discretization_level.

        Args:
            node: An internal node of the tree, for which to compute the
                posterior log joint.

        Returns:
            log P(node time = t, character states, tree topology) for t in
                [0, T], where T is the discretization_level.
        """
        if node not in self._posteriors.keys():
            raise ValueError(f"Unexistent node: {node}")

        return self._log_joints[node].copy()

    def posterior_time(self, node: str) -> np.array:
        """
        The posterior distribution of the time for a node.

        The posterior time distribution of a node, numerically computed, i.e.:
        P(node time = t | character states, tree topology) for t in [0, T]
        where T is the discretization_level.

        Args:
            node: An internal node of the CassiopeiaTree.

        Returns:
            P(node time = t | character states, tree topology) for t in [0, T]
                where T is the discretization_level.

        Raises:
            ValueError if the node is not an internal node.
        """
        if node not in self._posteriors.keys():
            raise ValueError(f"Unexistent node: {node}")

        return self._posteriors[node].copy()

    @property
    def mutation_rate(self):
        """
        The CRISPR/Cas9 mutation rate.
        """
        return self._mutation_rate

    @property
    def birth_rate(self):
        """
        The phylogeny birth rate.
        """
        return self._birth_rate

    @property
    def discretization_level(self):
        """
        How many timesteps are used to discretize time.
        """
        return self._discretization_level

    @property
    def sampling_probability(self):
        """
        The probability that a leaf in the ground truth tree was sampled.
        """
        return self._sampling_probability
