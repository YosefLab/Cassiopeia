"""
This file stores a subclass of BranchLengthEstimator, the
IIDExponentialBayesian. Briefly, this model assumes that CRISPR/Cas9 mutates
each site independently and identically, with an exponential waiting time;
and that the phylogeny follows a subsampled Birth Process. Conditional on the
observed topology and on the character data, the posterior mean branch lengths
are used as branch length estimates.
"""
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from scipy import integrate
from scipy.special import binom

from cassiopeia.data import CassiopeiaTree

from .BranchLengthEstimator import BranchLengthEstimator
from ._iid_exponential_bayesian import PyDP


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
        verbose: Verbosity level.

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
        verbose: bool = False,
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
        self._verbose = verbose
        self._log_likelihood = None

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """
        Posterior mean under a Birth Process and IID memoryless mutations.

        The only caveat is that this method raises a ValueError if the
        discretization_size is too small.

        Raises:
            ValueError if discretization_size is too small.
        """
        self._validate_input_tree(tree)

        verbose = self._verbose

        self._precompute_Ks(tree)
        self._down_cache = {}  # type: Dict[Tuple[str, int, int], float]
        self._up_cache = {}  # type: Dict[Tuple[str, int, int], float]
        self._log_joints = {}  # type: Dict[str, np.array]
        self._posterior_means = {}  # type: Dict[str, float]
        self._posteriors = {}  # type: Dict[str, np.array]
        self._tree = tree

        time_start = time.time()
        self._populate_attributes_with_cpp_implementation()
        time_end = time.time()
        if verbose:
            print(f"time_cpp = {time_end - time_start}")

        time_start = time.time()
        self._populate_branch_lengths()
        time_end = time.time()
        if verbose:
            print(f"time_populate_branch_lengths = {time_end - time_start}")

    def _up(self, v, t, x) -> float:
        """
        Up log probability.

        The probability of generating all data above node v, and having node v
        be in state x and divide at time t. Recall that the process is
        discretized, so technically this is a probability mass, not a
        probability density. (Upon suitable normalization by a power of dt, we
        recover the density).
        """
        if (v, t, x) in self._up_cache:
            return self._up_cache[(v, t, x)]
        else:
            return -np.inf

    def _down(self, v, t, x) -> float:
        """
        Down log probability.

        The probability of generating all data at and below node v, starting
        from the branch of v at state x and time t.
        """
        if (v, t, x) in self._down_cache:
            return self._down_cache[(v, t, x)]
        else:
            return -np.inf

    def _populate_branch_lengths(self):
        tree = self._tree
        times = {}
        for node in self._non_root_internal_nodes(tree):
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
            raise ValueError("The root of the tree should have degree exactly "
                             "1.")
        for node in tree.internal_nodes:
            if node != tree.root and len(tree.children(node)) != 2:
                raise ValueError("Each internal node (different from the root)"
                                 " should have degree exactly 2.")

    def _populate_attributes_with_cpp_implementation(self):
        """
        A cpp implementation is run to compute the up and down caches,
        which is the computational bottleneck. The other attributes such as the
        log-likelihood, and the posteriors, are also populated because
        even these trivial computations are too slow in vanilla python.
        Looking forward, a cython implementation will hopefully be the
        best way forward.
        To remove anything that has to do with the cpp implementation, you just
        have to remove this function (and the gates around it).
        I.e., this python implementation is loosely coupled to the cpp call: we
        just have to remove the call to this method to turn it off, and all
        other code will work just fine. This is because all that this method
        does is *warm up the cache* with values computed from the cpp
        subprocess, and the caching process is totally transparent to the
        other methods of the class.
        """
        # First extract the relevant information from the tree and serialize it.
        tree = self._tree
        node_to_id = dict(zip(tree.nodes, range(len(tree.nodes))))
        id_to_node = dict(zip(range(len(tree.nodes)), tree.nodes))

        N = len(tree.nodes)

        children = sorted(
            [
                [node_to_id[v]] + [node_to_id[c] for c in tree.children(v)]
                for v in tree.nodes
            ]
        )
        children = [vec[1:] for vec in children]
        # print(f"children = {children}")
        # assert(False)

        root = node_to_id[tree.root]

        is_internal_node = np.array(
            sorted(
                [
                    [node_to_id[v], 1 * tree.is_internal_node(v)]
                    for v in tree.nodes
                ]
            )
        )[:, 1]
        # print(f"is_internal_node = {is_internal_node}")
        # assert(False)

        get_number_of_mutated_characters_in_node = np.array(
            sorted(
                [
                    [
                        node_to_id[v],
                        self._get_number_of_mutated_characters_in_node(tree, v),
                    ]
                    for v in tree.nodes
                ]
            )
        )[:, 1]
        # print(f"get_number_of_mutated_characters_in_node = {get_number_of_mutated_characters_in_node}")
        # assert(False)

        non_root_internal_nodes = [
            node_to_id[v] for v in self._non_root_internal_nodes(tree)
        ]
        # print(f"non_root_internal_nodes = {non_root_internal_nodes}")
        # assert(False)

        leaves = [node_to_id[v] for v in tree.leaves]
        # print(f"leaves = {leaves}")
        # assert(False)

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
        # print(f"parent = {parent}")
        # assert(False)

        K = tree.n_character
        # print(f"K = {K}")
        # assert(False)

        Ks = np.array(
            sorted([[node_to_id[v], self._Ks[v]] for v in tree.nodes])
        )[:, 1]
        # print(f"Ks = {Ks}")
        # assert(False)

        T = self._discretization_level
        # print(f"T = {T}")
        # assert(False)

        r = self._mutation_rate
        # print(f"r = {r}")
        # assert(False)

        lam = self._birth_rate
        # print(f"lam = {lam}")
        # assert(False)

        sampling_probability = self._sampling_probability
        # print(f"sampling_probability = {sampling_probability}")
        # assert(False)

        is_leaf = np.array(
            sorted([[node_to_id[v], 1 * tree.is_leaf(v)] for v in tree.nodes])
        )[:, 1]
        # print(f"is_leaf = {is_leaf}")
        # assert(False)

        dp = PyDP()
        dp.run(
            N=N,
            children=children,
            root=root,
            is_internal_node=is_internal_node,
            get_number_of_mutated_characters_in_node=get_number_of_mutated_characters_in_node,
            non_root_internal_nodes=non_root_internal_nodes,
            leaves=leaves,
            parent=parent,
            K=K,
            Ks=Ks,
            T=T,
            r=r,
            lam=lam,
            sampling_probability=sampling_probability,
            is_leaf=is_leaf,
        )

        # Map back to strings
        for key_value in dp.get_down_res():
            assert len(key_value) == 2
            key = key_value[0]
            assert len(key) == 3
            value = key_value[1]
            self._down_cache[(id_to_node[key[0]], key[1], key[2])] = value
        # print(f"self._down_cache = {self._down_cache}")
        # assert(False)

        for key_value in dp.get_up_res():
            assert len(key_value) == 2
            key = key_value[0]
            assert len(key) == 3
            value = key_value[1]
            self._up_cache[(id_to_node[key[0]], key[1], key[2])] = value
        # print(f"self._up_cache = {self._up_cache}")
        # assert(False)

        self._log_likelihood = dp.get_log_likelihood_res()
        # print(f"self._log_likelihood = {self._log_likelihood}")
        # assert(False)

        for key_value in dp.get_log_joints_res():
            assert len(key_value) == 2
            key = key_value[0]
            value = key_value[1]
            assert len(value) == self._discretization_level + 1
            self._log_joints[id_to_node[key]] = np.array(value)
        # print(f"self._log_joints = {self._log_joints}")
        # assert(False)

        for key_value in dp.get_posteriors_res():
            assert len(key_value) == 2
            key = key_value[0]
            value = key_value[1]
            assert len(value) == self._discretization_level + 1
            self._posteriors[id_to_node[key]] = np.array(value)
        # print(f"self._posteriors = {self._posteriors}")
        # assert(False)

        for key_value in dp.get_posterior_means_res():
            assert len(key_value) == 2
            key = key_value[0]
            value = key_value[1]
            self._posterior_means[id_to_node[key]] = np.array(value)
        # print(f"self._posterior_means = {self._posterior_means}")
        # assert(False)

    def _get_number_of_mutated_characters_in_node(
        self, tree: CassiopeiaTree, v: str
    ):
        states = tree.get_character_states(v)
        return len(
            [s for s in states if s != 0 and s != tree.missing_state_indicator]
        )

    @classmethod
    def _non_root_internal_nodes(cls, tree: CassiopeiaTree) -> List[str]:
        """Returns internal nodes in tree (excluding the root).

        Returns:
            The internal nodes of the tree that are not the root (i.e. all
            nodes not at the leaves, and not the root)
        """
        return list(set(tree.nodes) - set(tree.root) - set(tree.leaves))

    def _precompute_Ks(self, tree: CassiopeiaTree):
        """
        For each vertex in the tree, we determine how many characters k are
        not considered missing starting here.
        """
        self._Ks = {}
        self._Ks[tree.root] = tree.n_character
        for (parent, child) in tree.edges:
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            k = 0
            for (parent_state, child_state) in zip(parent_states, child_states):
                if (
                    parent_state != 0
                    and parent_state != tree.missing_state_indicator
                ):
                    # Is an already-existing mutation
                    k += 1
                elif (
                    parent_state == 0
                    and child_state != tree.missing_state_indicator
                ):
                    # Is an unmutated character that did not go missing so we should track.
                    k += 1
                # Check that imputable missing states have been imputed.
                if (
                    parent_state != 0
                    and parent_state != tree.missing_state_indicator
                    and child_state == tree.missing_state_indicator
                ):
                    raise ValueError(
                        "Imputable missing states should have been imputed first. "
                        "Use the CassiopeiaTree.reconstruct_ancestral_characters method to do this."
                    )
            self._Ks[child] = k
        # Check monotonicity of Ks
        for (parent, child) in tree.edges:
            assert self._Ks[parent] >= self._Ks[child]

    @property
    def log_likelihood(self):
        """
        The log-likelihood of the training data under the estimated model.
        """
        return self._log_likelihood

    def log_joints(self, node: str) -> np.array:
        """
        log joint note time probabilities.

        The log joint probability density of the observed tree topology, state
        vectors, and all possible times of a node in the tree. In other words:
        log P(node time = t, character states, tree topology) for t in [0, T]
        where T is the discretization_level.

        Args:
            node: An internal node of the tree, for which to compute the
                posterior log joint.

        Returns:
            log P(node time = t, character states, tree topology) for t in [0, T]
            where T is the discretization_level.
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

    @classmethod
    def exact_log_full_joint(
        self,
        tree: CassiopeiaTree,
        mutation_rate: float,
        birth_rate: float,
        sampling_probability: float,
    ) -> float:
        """
        Exact log full joint probability computation.

        This method is used for testing the implementation of the model.

        The log full joint probability density of the observed tree topology,
        state vectors, and branch lengths. In other words:
        log P(branch lengths, character states, tree topology)
        Intergrating this function allows computing the marginals and hence
        the posteriors of the times of any internal node in the tree.

        Note that this method is only fast enough for small trees. It's
        run time scales exponentially with the number of internal nodes of the
        tree.

        Args:
            tree: The CassiopeiaTree containing the tree topology and all
                character states.
            node: An internal node of the tree, for which to compute the
                posterior log joint.
            mutation_rate: The mutation rate of the model.
            birth_rate: The birth rate of the model.
            sampling_probability: The sampling probability of the model.

        Returns:
            log P(branch lengths, character states, tree topology)
        """
        tree = deepcopy(tree)
        ll = 0.0
        lam = birth_rate
        r = mutation_rate
        p = sampling_probability
        q_inv = (1.0 - p) / p
        lg = np.log
        e = np.exp
        b = binom
        T = tree.get_max_depth_of_tree()
        for (p, c) in tree.edges:
            t = tree.get_branch_length(p, c)
            # Birth process with subsampling likelihood
            h = T - tree.get_time(p)
            h_tilde = T - tree.get_time(c)
            if c in tree.leaves:
                # "Easy" case
                assert h_tilde == 0
                ll += (
                    2.0 * lg(q_inv + 1.0)
                    + lam * h
                    - 2.0 * lg(q_inv + e(lam * h))
                    + lg(sampling_probability)
                )
            else:
                ll += (
                    lg(lam)
                    + lam * h
                    - 2.0 * lg(q_inv + e(lam * h))
                    + 2.0 * lg(q_inv + e(lam * h_tilde))
                    - lam * h_tilde
                )
            # Mutation process likelihood
            cuts = len(
                tree.get_mutations_along_edge(
                    p, c, treat_missing_as_mutations=False
                )
            )
            uncuts = tree.get_character_states(c).count(0)
            # Care must be taken here, we might get a nan
            if np.isnan(lg(1 - e(-t * r)) * cuts):
                return -np.inf
            ll += (
                (-t * r) * uncuts
                + lg(1 - e(-t * r)) * cuts
                + lg(b(cuts + uncuts, cuts))
            )
        return ll

    @classmethod
    def numerical_log_likelihood(
        self,
        tree: CassiopeiaTree,
        mutation_rate: float,
        birth_rate: float,
        sampling_probability: float,
        epsrel: float = 0.01,
    ) -> np.array:
        """
        Numerical log likelihood of the observed data.

        This method is used for testing the implementation of log_likelihood of
        the model.

        The log likelihood of the observed tree topology and state vectors,
        i.e.:
        log P(character states, tree topology)

        Note that this method is only fast enough for small trees. Its
        run time scales exponentially with the number of internal nodes of the
        tree.

        Args:
            tree: The CassiopeiaTree containing the tree topology and all
                character states.
            mutation_rate: The mutation rate of the model.
            birth_rate: The birth rate of the model.
            sampling_probability: The sampling probability of the model.
            epsrel: The degree of tolerance for the numerical integrals
                performed.

        Returns:
            log P(character states, tree topology)
        """
        tree = deepcopy(tree)

        def f(*args):
            times_list = args
            times = {}
            for node, t in list(
                zip(self._non_root_internal_nodes(tree), times_list)
            ):
                times[node] = t
            times[tree.root] = 0
            for leaf in tree.leaves:
                times[leaf] = 1.0
            for (p, c) in tree.edges:
                if times[p] >= times[c]:
                    return 0.0
            tree.set_times(times)
            return np.exp(
                IIDExponentialBayesian.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                )
            )

        res = np.log(
            integrate.nquad(
                f,
                [[0, 1]] * len(self._non_root_internal_nodes(tree)),
                opts={"epsrel": epsrel},
            )[0]
        )
        assert not np.isnan(res)
        return res

    @classmethod
    def numerical_log_joints(
        self,
        tree: CassiopeiaTree,
        node: str,
        mutation_rate: float,
        birth_rate: float,
        sampling_probability: float,
        discretization_level: int,
        epsrel: float = 0.01,
    ) -> np.array:
        """
        Numerical log joint probability computation.

        This method is used for testing the implementation of log_joints of the
        model.

        The log joint probability density of the observed tree topology, state
        vectors, and all possible times of a node in the tree. In other words:
        log P(node time = t, character states, tree topology) for t in [0, T]
        where T is the discretization_level.

        Note that this method is only fast enough for small trees. It's
        run time scales exponentially with the number of internal nodes of the
        tree.

        Args:
            tree: The CassiopeiaTree containing the tree topology and all
                character states.
            node: An internal node of the tree, for which to compute the
                posterior log joint.
            mutation_rate: The mutation rate of the model.
            birth_rate: The birth rate of the model.
            sampling_probability: The sampling probability of the model.
            discretization_level: The number of timesteps used to discretize
                time. The output thus is a vector of length
                discretization_level + 1.
            epsrel: The degree of tolerance for the numerical integrals
                performed.

        Returns:
            log P(node time = t, character states, tree topology) for t in [0, T]
            where T is the discretization_level.
        """
        res = np.zeros(shape=(discretization_level + 1,))
        other_nodes = [
            n for n in self._non_root_internal_nodes(tree) if n != node
        ]
        node_time = -1

        tree = deepcopy(tree)

        def f(*args):
            times_list = args
            times = {}
            times[node] = node_time
            assert len(other_nodes) == len(times_list)
            for other_node, t in list(zip(other_nodes, times_list)):
                times[other_node] = t
            times[tree.root] = 0
            for leaf in tree.leaves:
                times[leaf] = 1.0
            for (p, c) in tree.edges:
                if times[p] >= times[c]:
                    return 0.0
            tree.set_times(times)
            return np.exp(
                IIDExponentialBayesian.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                )
            )

        for i in range(discretization_level + 1):
            node_time = i / discretization_level
            if len(other_nodes) == 0:
                # There is nothing to integrate over.
                times = {}
                times[tree.root] = 0
                for leaf in tree.leaves:
                    times[leaf] = 1.0
                times[node] = node_time
                tree.set_times(times)
                res[i] = self.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    sampling_probability=sampling_probability,
                )
                res[i] -= np.log(discretization_level)
            else:
                res[i] = (
                    np.log(
                        integrate.nquad(
                            f,
                            [[0, 1]]
                            * (len(self._non_root_internal_nodes(tree)) - 1),
                            opts={"epsrel": epsrel},
                        )[0]
                    )
                    - np.log(discretization_level)
                )
                assert not np.isnan(res[i])

        return res

    @classmethod
    def numerical_posterior_time(
        self,
        tree: CassiopeiaTree,
        node: str,
        mutation_rate: float,
        birth_rate: float,
        sampling_probability: float,
        discretization_level: int,
        epsrel: float = 0.01,
    ) -> np.array:
        """
        Numerical posterior time inference under the model.

        This method is used for testing the implementation of posterior_time of
        the model.

        The posterior time distribution of a node, numerically computed, i.e.:
        P(node time = t | character states, tree topology) for t in [0, T]
        where T is the discretization_level.

        Note that this method is only fast enough for small trees. It's
        run time scales exponentially with the number of internal nodes of the
        tree.

        Args:
            tree: The CassiopeiaTree containing the tree topology and all
                character states.
            node: An internal node of the tree, for which to compute the
                posterior time distribution.
            mutation_rate: The mutation rate of the model.
            birth_rate: The birth rate of the model.
            sampling_probability: The sampling probability of the model.
            discretization_level: The number of timesteps used to discretize
                time. The output thus is a vector of length
                discretization_level + 1.
            epsrel: The degree of tolerance for the numerical integrals
                performed.

        Returns:
            P(node time = t | character states, tree topology) for t in [0, T]
            where T is the discretization_level.
        """
        numerical_log_joints = self.numerical_log_joints(
            tree=tree,
            node=node,
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            epsrel=epsrel,
        )
        numerical_posterior = np.exp(
            numerical_log_joints - numerical_log_joints.max()
        )
        numerical_posterior /= numerical_posterior.sum()
        return numerical_posterior.copy()

    def _simple_inference_sanity_check(self):
        """
        Tests that the likelihood computed from each leaf node is correct.
        """
        tree = self._tree
        for leaf in tree.leaves:
            model_log_likelihood_up = (
                self._up(
                    leaf,
                    self._discretization_level,
                    self._get_number_of_mutated_characters_in_node(tree, leaf),
                )
                - np.log(self._birth_rate * 1.0 / self._discretization_level)
                + np.log(self._sampling_probability)
            )
            np.testing.assert_approx_equal(
                self._log_likelihood, model_log_likelihood_up, significant=2
            )
