import multiprocessing
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import binom, logsumexp

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from cassiopeia.data import CassiopeiaTree

from . import utils
from .BranchLengthEstimator import (
    BranchLengthEstimator,
    BranchLengthEstimatorError,
)


class IIDExponentialPosteriorMeanBLE(BranchLengthEstimator):
    r"""
    TODO: Update to match my technical write-up.

    Same model as IIDExponentialBLE but computes the posterior mean instead
    of the MLE. The phylogeny model is chosen to be a birth process.

    This estimator requires that the ancestral states are provided.

    TODO: Use numpy autograd to optimize the hyperparams? (Empirical Bayes)

    We compute the posterior means using a forward-backward-style algorithm
    (DP on a tree).

    Args: TODO

    Attributes: TODO

    """

    def __init__(
        self,
        mutation_rate: float,
        birth_rate: float,
        discretization_level: int,
        sampling_probability: float = 1.0,
        enforce_parsimony: bool = True,
        use_cpp_implementation: bool = False,
        debug_cpp_implementation: bool = False,
        verbose: bool = False,
    ) -> None:
        # TODO: If we use autograd, we can tune the hyperparams with gradient
        # descent?
        self.mutation_rate = mutation_rate
        # TODO: Is there some easy heuristic way to set this to a reasonable
        # value and thus avoid grid searching it / optimizing it?
        self.birth_rate = birth_rate
        if sampling_probability <= 0 or sampling_probability > 1:
            raise BranchLengthEstimatorError(
                "sampling_probability should be in (0, 1]. "
                f"{sampling_probability} provided."
            )
        self.sampling_probability = sampling_probability
        self.discretization_level = discretization_level
        self.enforce_parsimony = enforce_parsimony
        self.use_cpp_implementation = use_cpp_implementation
        self.debug_cpp_implementation = debug_cpp_implementation
        self.verbose = verbose

    def _compute_log_likelihood(self):
        tree = self.tree
        log_likelihood = 0
        # TODO: Should I also add a division event when the root has multiple
        # children? (If not, the joint we are computing won't integrate to 1;
        # on the other hand, this is a constant multiplicative term that doesn't
        # affect inference.
        for child_of_root in tree.children(tree.root):
            log_likelihood += self.down(child_of_root, 0, 0)
        self.log_likelihood = log_likelihood

    def _compute_log_joint(self, v, t):
        r"""
        P(t_v = t, X, T).
        Depending on whether we are enforcing parsimony or not, we consider
        different possible number of cuts for v.
        """
        tree = self.tree
        assert tree.is_internal_node(v) and v != tree.root
        enforce_parsimony = self.enforce_parsimony
        children = tree.children(v)
        if enforce_parsimony:
            valid_num_cuts = [tree.get_number_of_mutated_characters_in_node(v)]
        else:
            valid_num_cuts = range(
                tree.get_number_of_mutated_characters_in_node(v) + 1
            )
        ll_for_xs = []
        for x in valid_num_cuts:
            ll_for_xs.append(
                sum([self.down(u, t, x) for u in children]) + self.up(v, t, x)
            )
        return logsumexp(ll_for_xs)

    def _compute_posteriors(self):
        tree = self.tree
        discretization_level = self.discretization_level
        log_joints = {}  # log P(t_v = t, X, T)
        posteriors = {}  # P(t_v = t | X, T)
        posterior_means = {}  # E[t_v = t | X, T]
        for v in tree.non_root_internal_nodes:
            # Compute the posterior for this node
            log_joint = np.zeros(shape=(discretization_level + 1,))
            for t in range(discretization_level + 1):
                log_joint[t] = self._compute_log_joint(v, t)
            log_joints[v] = log_joint.copy()
            posterior = np.exp(log_joint - log_joint.max())
            posterior /= np.sum(posterior)
            posteriors[v] = posterior
            posterior_means[v] = (
                posterior * np.array(range(discretization_level + 1))
            ).sum() / discretization_level
        self.log_joints = log_joints
        self.posteriors = posteriors
        self.posterior_means = posterior_means

    def _populate_branch_lengths(self):
        tree = self.tree
        posterior_means = self.posterior_means
        times = {}
        for node in tree.non_root_internal_nodes:
            times[node] = posterior_means[node]
        times[tree.root] = 0.0
        for leaf in tree.leaves:
            times[leaf] = 1.0
        tree.set_times(times)

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        self._precompute_p_unsampled()
        self._down_cache = {}
        self._up_cache = {}
        self.tree = tree
        verbose = self.verbose
        if self.debug_cpp_implementation:
            # Write out true dp values to check by eye against c++
            # implementation values.
            self._write_out_dps()
        if self.use_cpp_implementation:
            time_cpp_start = time.time()
            if self.debug_cpp_implementation:
                # Use a directory that won't go away.
                self._populate_attributes_with_cpp_implementation(
                    tmp_dir=os.getcwd() + "/tmp"
                )
            else:
                # Use a temporary directory.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    self._populate_attributes_with_cpp_implementation(tmp_dir)
            time_cpp_end = time.time()
            if verbose:
                print(f"time_cpp = {time_cpp_end - time_cpp_start}")
        else:
            time_compute_log_likelihood_start = time.time()
            self._compute_log_likelihood()
            time_compute_log_likelihood_end = time.time()
            if verbose:
                print(
                    f"time_compute_log_likelihood (dp_down) = {time_compute_log_likelihood_end - time_compute_log_likelihood_start}"
                )
            time_compute_posteriors_start = time.time()
            self._compute_posteriors()
            time_compute_posteriors_end = time.time()
            if verbose:
                print(
                    f"time_compute_posteriors (dp_up) = {time_compute_posteriors_end - time_compute_posteriors_start}"
                )
        time_populate_branch_lengths_start = time.time()
        self._populate_branch_lengths()
        time_populate_branch_lengths_end = time.time()
        if verbose:
            print(
                f"time_populate_branch_lengths = {time_populate_branch_lengths_end - time_populate_branch_lengths_start}"
            )

    def _precompute_p_unsampled(self):
        discretization_level = self.discretization_level
        sampling_probability = self.sampling_probability
        lam = self.birth_rate
        dt = 1.0 / discretization_level
        if 1 - lam * dt <= 0:
            raise ValueError(f"1 - lam * dt = 1 - {lam} * {dt} should be positive!")
        p_unsampled = [-np.inf for i in range(discretization_level + 1)]
        if sampling_probability < 1.0:
            p_unsampled[discretization_level] = np.log(1.0 - sampling_probability)
            for t in range(discretization_level - 1, -1, -1):
                log_likelihoods_cases = [
                    np.log(1 - lam * dt) + p_unsampled[t + 1],  # Nothing happens
                    np.log(lam * dt) + 2 * p_unsampled[t + 1]  # Cell division event
                ]
                p_unsampled[t] = logsumexp(log_likelihoods_cases)
        self._p_unsampled = p_unsampled

    def _write_out_dps(self):
        r"""
        For debugging the c++ implementation:
        This writes out the down and up values of the correct python
        implementation to the files tmp/down_true.txt
        and
        tmp/up_true.txt
        respectively.
        Compare these against tmp/down.txt and tmp/up.txt, which are the values
        computed by the c++ implementation.
        """
        tree = self.tree
        N = len(tree.nodes)
        T = self.discretization_level
        K = tree.n_character
        id_to_node = dict(zip(range(len(tree.nodes)), tree.nodes))

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        res = ""
        for v_id in range(N):
            v = id_to_node[v_id]
            if v == tree.root:
                continue
            for t in range(T + 1):
                for x in range(K + 1):
                    if self._state_is_valid(v, t, x):
                        res += (
                            str(v_id)
                            + " "
                            + str(t)
                            + " "
                            + str(x)
                            + " "
                            + str(self.down(v, t, x))
                            + "\n"
                        )
        with open("tmp/down_true.txt", "w") as fout:
            fout.write(res)

        res = ""
        for v_id in range(N):
            v = id_to_node[v_id]
            for t in range(T + 1):
                for x in range(K + 1):
                    if self._state_is_valid(v, t, x):
                        res += (
                            str(v_id)
                            + " "
                            + str(t)
                            + " "
                            + str(x)
                            + " "
                            + str(self.up(v, t, x))
                            + "\n"
                        )
        with open("tmp/up_true.txt", "w") as fout:
            fout.write(res)

    def _write_out_list_of_lists(self, lls: List[List[int]], filename: str):
        res = ""
        for l in lls:
            for i, x in enumerate(l):
                if i:
                    res += " "
                res += str(x)
            res += "\n"
        with open(filename, "w") as file:
            file.write(res)

    def _populate_attributes_with_cpp_implementation(self, tmp_dir):
        r"""
        A cpp implementation is run to compute up and down caches, which is
        the computational bottleneck. The other attributes such as the
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
        tree = self.tree
        node_to_id = dict(zip(tree.nodes, range(len(tree.nodes))))
        id_to_node = dict(zip(range(len(tree.nodes)), tree.nodes))

        N = [[len(tree.nodes)]]
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        self._write_out_list_of_lists(N, f"{tmp_dir}/N.txt")

        children = [
            [node_to_id[v]]
            + [len(tree.children(v))]
            + [node_to_id[c] for c in tree.children(v)]
            for v in tree.nodes
        ]
        self._write_out_list_of_lists(children, f"{tmp_dir}/children.txt")

        root = [[node_to_id[tree.root]]]
        self._write_out_list_of_lists(root, f"{tmp_dir}/root.txt")

        is_internal_node = [
            [node_to_id[v], 1 * tree.is_internal_node(v)] for v in tree.nodes
        ]
        self._write_out_list_of_lists(
            is_internal_node, f"{tmp_dir}/is_internal_node.txt"
        )

        get_number_of_mutated_characters_in_node = [
            [node_to_id[v], tree.get_number_of_mutated_characters_in_node(v)]
            for v in tree.nodes
        ]
        self._write_out_list_of_lists(
            get_number_of_mutated_characters_in_node,
            f"{tmp_dir}/get_number_of_mutated_characters_in_node.txt",
        )

        non_root_internal_nodes = [
            [node_to_id[v]] for v in tree.non_root_internal_nodes
        ]
        self._write_out_list_of_lists(
            non_root_internal_nodes, f"{tmp_dir}/non_root_internal_nodes.txt"
        )

        leaves = [[node_to_id[v]] for v in tree.leaves]
        self._write_out_list_of_lists(leaves, f"{tmp_dir}/leaves.txt")

        parent = [
            [node_to_id[v], node_to_id[tree.parent(v)]]
            for v in tree.nodes
            if v != tree.root
        ]
        self._write_out_list_of_lists(parent, f"{tmp_dir}/parent.txt")

        K = [[tree.n_character]]
        self._write_out_list_of_lists(K, f"{tmp_dir}/K.txt")

        T = [[self.discretization_level]]
        self._write_out_list_of_lists(T, f"{tmp_dir}/T.txt")

        enforce_parsimony = [[1 * self.enforce_parsimony]]
        self._write_out_list_of_lists(
            enforce_parsimony, f"{tmp_dir}/enforce_parsimony.txt"
        )

        r = [[self.mutation_rate]]
        self._write_out_list_of_lists(r, f"{tmp_dir}/r.txt")

        lam = [[self.birth_rate]]
        self._write_out_list_of_lists(lam, f"{tmp_dir}/lam.txt")

        sampling_probability = [[self.sampling_probability]]
        self._write_out_list_of_lists(sampling_probability, f"{tmp_dir}/sampling_probability.txt")

        is_leaf = [[node_to_id[v], 1 * tree.is_leaf(v)] for v in tree.nodes]
        self._write_out_list_of_lists(is_leaf, f"{tmp_dir}/is_leaf.txt")

        # Run the c++ implementation
        try:
            # os.system('IIDExponentialPosteriorMeanBLE')
            subprocess.run(
                [
                    "./IIDExponentialPosteriorMeanBLE",
                    f"{tmp_dir}",
                    f"{tmp_dir}",
                ],
                check=True,
                cwd=os.path.dirname(__file__),
            )
        except subprocess.CalledProcessError:
            raise BranchLengthEstimatorError(
                "Couldn't run c++ implementation,"
                " or c++ implementation started running and errored."
            )

        # Load the c++ implementation results into the cache
        with open(f"{tmp_dir}/down.txt", "r") as fin:
            for line in fin:
                v, t, x, ll = line.split(" ")
                v, t, x = int(v), int(t), int(x)
                ll = float(ll)
                self._down_cache[(id_to_node[v], t, x)] = ll
        with open(f"{tmp_dir}/up.txt", "r") as fin:
            for line in fin:
                v, t, x, ll = line.split(" ")
                v, t, x = int(v), int(t), int(x)
                ll = float(ll)
                self._up_cache[(id_to_node[v], t, x)] = ll

        discretization_level = self.discretization_level

        # Load the log_likelihood
        with open(f"{tmp_dir}/log_likelihood.txt", "r") as fin:
            self.log_likelihood = float(fin.read())

        # Load the posteriors
        log_joints = {}  # log P(t_v = t, X, T)
        with open(f"{tmp_dir}/log_joints.txt", "r") as fin:
            for line in fin:
                vals = line.split(" ")
                assert len(vals) == discretization_level + 2
                v_id = int(vals[0])
                log_joint = np.zeros(shape=(discretization_level + 1,))
                for i, val in enumerate(vals[1:]):
                    log_joint[i] = float(val)
                log_joints[id_to_node[v_id]] = log_joint

        posteriors = {}  # P(t_v = t | X, T)
        with open(f"{tmp_dir}/posteriors.txt", "r") as fin:
            for line in fin:
                vals = line.split(" ")
                assert len(vals) == discretization_level + 2
                v_id = int(vals[0])
                posterior = np.zeros(shape=(discretization_level + 1,))
                for i, val in enumerate(vals[1:]):
                    posterior[i] = float(val)
                posteriors[id_to_node[v_id]] = posterior

        posterior_means = {}  # E[t_v = t | X, T]
        with open(f"{tmp_dir}/posterior_means.txt", "r") as fin:
            for line in fin:
                v_id, val = line.split(" ")
                v_id = int(v_id)
                val = float(val)
                posterior_means[id_to_node[v_id]] = val

        self.log_joints = log_joints
        self.posteriors = posteriors
        self.posterior_means = posterior_means

    def _compatible_with_observed_data(self, x, observed_cuts) -> bool:
        if self.enforce_parsimony:
            return x == observed_cuts
        else:
            return x <= observed_cuts

    def _state_is_valid(self, v, t, x) -> bool:
        r"""
        Used to optimize the DP by avoiding states with 0 probability.
        The number of mutations should be between those of v and its parent.
        """
        tree = self.tree
        if v == tree.root:
            return x == 0
        p = tree.parent(v)
        cuts_v = tree.get_number_of_mutated_characters_in_node(v)
        cuts_p = tree.get_number_of_mutated_characters_in_node(p)
        if self.enforce_parsimony:
            return cuts_p <= x <= cuts_v
        else:
            return x <= cuts_v

    def up(self, v, t, x) -> float:
        r"""
        TODO: Rename this _up?
        log P(X_up(b(v)), T_up(b(v)), t \in t_b(v), X_b(v)(t) = x)
        TODO: Update to match my technical write-up.
        """
        # Avoid doing anything at all for invalid states.
        if not self._state_is_valid(v, t, x):
            return -np.inf
        if (v, t, x) in self._up_cache:  # TODO: Use arrays?
            # TODO: Use a decorator instead of a hand-made cache?
            return self._up_cache[(v, t, x)]
        if self.use_cpp_implementation and not self.debug_cpp_implementation:
            raise ValueError(
                f"Bug in cpp implementation: State up({(v, t, x)})"
                f" was not populated."
            )
        # Pull out params
        r = self.mutation_rate
        lam = self.birth_rate
        dt = 1.0 / self.discretization_level
        K = self.tree.n_character
        tree = self.tree
        assert 0 <= t <= self.discretization_level
        assert 0 <= x <= K
        if not (1.0 - lam * dt - K * r * dt > 0):
            raise ValueError("Please choose a bigger discretization_level.")
        log_likelihood = 0.0
        if v == tree.root:  # Base case: we reached the root of the tree.
            if t == 0 and x == tree.get_number_of_mutated_characters_in_node(v):
                log_likelihood = 0.0
            else:
                log_likelihood = -np.inf
        elif t == 0:
            # Base case: we reached the start of the process, but we're not yet
            # at the root.
            assert v != tree.root
            log_likelihood = -np.inf
        else:  # Recursion.
            log_likelihoods_cases = []
            # Case 1: Nothing happened
            log_likelihoods_cases.append(
                np.log(1.0 - lam * dt - (K - x) * r * dt) + self.up(v, t - 1, x)
            )
            # Case 2: Mutation happened
            if x - 1 >= 0:
                log_likelihoods_cases.append(
                    np.log((K - (x - 1)) * r * dt) + self.up(v, t - 1, x - 1)
                )
            # Case 3: A cell division happened AND both lineages persisted
            if v != tree.root:
                p = tree.parent(v)
                if self._compatible_with_observed_data(
                    x,
                    tree.get_number_of_mutated_characters_in_node(
                        p
                    ),  # If we want to ignore missing data, we just have to replace x by x-gone_missing(p->v). I.e. dropped out characters become free mutations.
                ):
                    siblings = [u for u in tree.children(p) if u != v]
                    ll = (
                        np.log(lam * dt)
                        + self.up(
                            p, t - 1, x
                        )  # If we want to ignore missing data, we just have to replace x by x-gone_missing(p->v). I.e. dropped out characters become free mutations.
                        + sum(
                            [self.down(u, t, x) for u in siblings]
                        )  # If we want to ignore missing data, we just have to replace x by cuts(p)+gone_missing(p->u). I.e. dropped out characters become free mutations.
                    )
                    log_likelihoods_cases.append(ll)
            # Case 4: There was a cell division event, BUT one of the two
            # lineages was not sampled
            ll = np.log(2 * lam * dt) + self._p_unsampled[t - 1] + self.up(v, t - 1, x)
            log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self._up_cache[(v, t, x)] = log_likelihood
        return log_likelihood

    def down(self, v, t, x) -> float:
        r"""
        TODO: Rename this _down?
        log P(X_down(v), T_down(v) | t_v = t, X_v = x)
        TODO: Update to match my technical write-up.
        """
        # Avoid doing anything at all for invalid states.
        if not self._state_is_valid(v, t, x):
            return -np.inf
        if (v, t, x) in self._down_cache:
            # TODO: Use a decorator instead of a hand-made cache?
            return self._down_cache[(v, t, x)]
        if self.use_cpp_implementation and not self.debug_cpp_implementation:
            raise ValueError(
                f"Bug in cpp implementation: State "
                f"down({(v, t, x)}) was not populated."
            )
        # Pull out params
        discretization_level = self.discretization_level
        r = self.mutation_rate
        lam = self.birth_rate
        dt = 1.0 / self.discretization_level
        K = self.tree.n_character
        tree = self.tree
        assert v != tree.root
        assert 0 <= t <= self.discretization_level
        assert 0 <= x <= K
        if not (1.0 - lam * dt - K * r * dt > 0):
            raise ValueError("Please choose a bigger discretization_level.")
        log_likelihood = 0.0
        if t == discretization_level:  # Base case
            if tree.is_leaf(
                v
            ) and x == tree.get_number_of_mutated_characters_in_node(v):
                log_likelihood = np.log(self.sampling_probability)
            else:
                log_likelihood = -np.inf
        else:  # Recursion.
            log_likelihoods_cases = []
            # Case 1: Nothing happens
            log_likelihoods_cases.append(
                np.log(1.0 - lam * dt - (K - x) * r * dt)
                + self.down(v, t + 1, x)
            )
            # Case 2: One character mutates.
            if x + 1 <= K:
                log_likelihoods_cases.append(
                    np.log((K - x) * r * dt) + self.down(v, t + 1, x + 1)
                )
            # Case 3: Cell divides AND both lineages persist.
            # The number of cuts at this state must match the ground truth.
            if self._compatible_with_observed_data(
                x, tree.get_number_of_mutated_characters_in_node(v)
            ) and not tree.is_leaf(v):
                ll = sum(
                    [
                        self.down(child, t + 1, x) for child in tree.children(v)
                    ]  # If we want to ignore missing data, we just have to replace x by x+gone_missing(p->v). I.e. dropped out characters become free mutations.
                ) + np.log(lam * dt)
                log_likelihoods_cases.append(ll)
            # Case 4: Cell divides, BUT one of the two
            # lineages is not sampled
            ll = np.log(2 * lam * dt) + self._p_unsampled[t + 1] + self.down(v, t + 1, x)
            log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self._down_cache[(v, t, x)] = log_likelihood
        return log_likelihood

    @classmethod
    def exact_log_full_joint(
        self, tree: CassiopeiaTree, mutation_rate: float, birth_rate: float
    ) -> float:
        r"""
        log P(T, X, branch_lengths), i.e. the full joint log likelihood given
        both character vectors _and_ branch lengths.
        """
        tree = deepcopy(tree)
        ll = 0.0
        lam = birth_rate
        r = mutation_rate
        lg = np.log
        e = np.exp
        b = binom
        for (p, c) in tree.edges:
            t = tree.get_branch_length(p, c)
            # Birth process likelihood
            ll += -t * lam
            if c not in tree.leaves:
                ll += lg(lam)
            # Mutation process likelihood
            cuts = tree.get_number_of_mutations_along_edge(p, c)
            uncuts = tree.get_number_of_unmutated_characters_in_node(c)
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
        epsrel: float = 0.01,
    ):
        r"""
        log P(T, X), i.e. the marginal log likelihood given _only_ tree
        topology and character vectors (including those of internal nodes).
        It is computed with a grid.
        """

        tree = deepcopy(tree)

        def f(*args):
            times_list = args
            times = {}
            for node, time in list(
                zip(tree.non_root_internal_nodes, times_list)
            ):
                times[node] = time
            times[tree.root] = 0
            for leaf in tree.leaves:
                times[leaf] = 1.0
            for (p, c) in tree.edges:
                if times[p] >= times[c]:
                    return 0.0
            tree.set_times(times)
            return np.exp(
                IIDExponentialPosteriorMeanBLE.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                )
            )

        res = np.log(
            integrate.nquad(
                f,
                [[0, 1]] * len(tree.non_root_internal_nodes),
                opts={"epsrel": epsrel},
            )[0]
        )
        assert not np.isnan(res)
        return res

    @classmethod
    def numerical_log_joint(
        self,
        tree: CassiopeiaTree,
        node,
        mutation_rate: float,
        birth_rate: float,
        discretization_level: int,
        epsrel: float = 0.01,
    ):
        r"""
        log P(t_node = t, X, T) for each t in the interval [0, 1] discretized
        to the level discretization_level
        """
        res = np.zeros(shape=(discretization_level + 1,))
        other_nodes = [n for n in tree.non_root_internal_nodes if n != node]
        node_time = -1

        tree = deepcopy(tree)

        def f(*args):
            times_list = args
            times = {}
            times[node] = node_time
            assert len(other_nodes) == len(times_list)
            for other_node, time in list(zip(other_nodes, times_list)):
                times[other_node] = time
            times[tree.root] = 0
            for leaf in tree.leaves:
                times[leaf] = 1.0
            for (p, c) in tree.edges:
                if times[p] >= times[c]:
                    return 0.0
            tree.set_times(times)
            return np.exp(
                IIDExponentialPosteriorMeanBLE.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
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
                )
                res[i] -= np.log(discretization_level)
            else:
                res[i] = (
                    np.log(
                        integrate.nquad(
                            f,
                            [[0, 1]] * (len(tree.non_root_internal_nodes) - 1),
                            opts={"epsrel": epsrel},
                        )[0]
                    )
                    - np.log(discretization_level)
                )
                assert not np.isnan(res[i])

        return res

    @classmethod
    def numerical_posterior(
        self,
        tree: CassiopeiaTree,
        node,
        mutation_rate: float,
        birth_rate: float,
        discretization_level: int,
        epsrel: float = 0.01,
    ):
        numerical_log_joint = self.numerical_log_joint(
            tree=tree,
            node=node,
            mutation_rate=mutation_rate,
            birth_rate=birth_rate,
            discretization_level=discretization_level,
            epsrel=epsrel,
        )
        numerical_posterior = np.exp(
            numerical_log_joint - numerical_log_joint.max()
        )
        numerical_posterior /= numerical_posterior.sum()
        return numerical_posterior


def _fit_model(model_and_tree):
    r"""
    This is used by IIDExponentialPosteriorMeanBLEGridSearchCV to
    parallelize the grid search. It must be defined here (at the top level of
    the module) for multiprocessing to be able to pickle it. (This is why
    coverage misses it)
    """
    model, tree = model_and_tree
    model.estimate_branch_lengths(tree)
    return model.log_likelihood


class IIDExponentialPosteriorMeanBLEGridSearchCV(BranchLengthEstimator):
    r"""
    Like IIDExponentialPosteriorMeanBLE but with automatic tuning of
    hyperparameters.

    This class fits the hyperparameters of IIDExponentialPosteriorMeanBLE based
    on data log-likelihood. I.e. is performs empirical Bayes.

    Args: TODO
    """

    def __init__(
        self,
        mutation_rates: Tuple[float] = (0,),
        birth_rates: Tuple[float] = (0,),
        sampling_probability: float = 1.0,
        discretization_level: int = 1000,
        enforce_parsimony: bool = True,
        use_cpp_implementation: bool = False,
        processes: int = 6,
        verbose: bool = False,
    ):
        self.mutation_rates = mutation_rates
        self.birth_rates = birth_rates
        self.sampling_probability = sampling_probability
        self.discretization_level = discretization_level
        self.enforce_parsimony = enforce_parsimony
        self.use_cpp_implementation = use_cpp_implementation
        self.processes = processes
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        mutation_rates = self.mutation_rates
        birth_rates = self.birth_rates
        sampling_probability = self.sampling_probability
        discretization_level = self.discretization_level
        enforce_parsimony = self.enforce_parsimony
        use_cpp_implementation = self.use_cpp_implementation
        processes = self.processes
        verbose = self.verbose

        lls = []
        grid = np.zeros(shape=(len(mutation_rates), len(birth_rates)))
        models = []
        mutation_and_birth_rates = []
        ijs = []
        for i, mutation_rate in enumerate(mutation_rates):
            for j, birth_rate in enumerate(birth_rates):
                if self.verbose:
                    print(
                        f"Fitting model with:\n"
                        f"mutation_rate={mutation_rate}\n"
                        f"birth_rate={birth_rate}"
                    )
                models.append(
                    IIDExponentialPosteriorMeanBLE(
                        mutation_rate=mutation_rate,
                        birth_rate=birth_rate,
                        sampling_probability=sampling_probability,
                        discretization_level=discretization_level,
                        enforce_parsimony=enforce_parsimony,
                        use_cpp_implementation=use_cpp_implementation,
                    )
                )
                mutation_and_birth_rates.append((mutation_rate, birth_rate))
                ijs.append((i, j))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            lls = list(
                map_fn(
                    _fit_model,
                    zip(models, [deepcopy(tree) for _ in range(len(models))]),
                )
            )
        lls_and_rates = list(zip(lls, mutation_and_birth_rates))
        for ll, (i, j) in list(zip(lls, ijs)):
            grid[i, j] = ll
        lls_and_rates.sort(reverse=True)
        (best_mutation_rate, best_birth_rate,) = lls_and_rates[
            0
        ][1]
        if verbose:
            print(
                f"Refitting model with:\n"
                f"best_mutation_rate={best_mutation_rate}\n"
                f"best_birth_rate={best_birth_rate}"
            )
        final_model = IIDExponentialPosteriorMeanBLE(
            mutation_rate=best_mutation_rate,
            birth_rate=best_birth_rate,
            sampling_probability=sampling_probability,
            discretization_level=discretization_level,
            enforce_parsimony=enforce_parsimony,
            use_cpp_implementation=use_cpp_implementation,
        )
        final_model.estimate_branch_lengths(tree)
        self.mutation_rate = best_mutation_rate
        self.birth_rate = best_birth_rate
        self.log_likelihood = final_model.log_likelihood
        self.log_joints = final_model.log_joints
        self.posteriors = final_model.posteriors
        self.posterior_means = final_model.posterior_means
        self.grid = grid

    def plot_grid(
        self, figure_file: Optional[str] = None, show_plot: bool = True
    ):
        utils.plot_grid(
            grid=self.grid,
            yticklabels=self.mutation_rates,
            xticklabels=self.birth_rates,
            ylabel=r"Mutation Rate ($r$)",
            xlabel=r"Birth Rate ($\lambda$)",
            title=f"Sampling Probability = {self.sampling_probability}\nLL = {self.log_likelihood}",
            figure_file=figure_file,
            show_plot=show_plot,
        )


class IIDExponentialPosteriorMeanBLEAutotune(BranchLengthEstimator):
    def __init__(
        self,
        discretization_level: int,
        enforce_parsimony: bool = True,
        use_cpp_implementation: bool = False,
        processes: int = 6,
        num_samples: int = 100,
        space: Optional[Dict] = None,
        search_alg=None,
        verbose: bool = False,
    ) -> None:
        self.discretization_level = discretization_level
        self.enforce_parsimony = enforce_parsimony
        self.use_cpp_implementation = use_cpp_implementation
        self.processes = processes
        self.num_samples = num_samples
        self.verbose = verbose
        if space is None:
            space = {
                "mutation_rate": tune.loguniform(0.1, 10.0),
                "birth_rate": tune.loguniform(0.1, 30.0),
                "sampling_probability": tune.loguniform(0.0000001, 1.0),
            }
        self.space = space
        if search_alg is None:
            search_alg = HyperOptSearch(
                metric="log_likelihood", mode="max"
            )
        self.search_alg = search_alg

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class.
        """
        self.tree = tree
        ray.init(num_cpus=self.processes)
        analysis = tune.run(
            self._trainable,
            config=self.space,
            num_samples=self.num_samples,
            search_alg=self.search_alg,
            metric='log_likelihood',
            mode='max'
        )
        ray.shutdown()
        self.analysis = analysis
        best_config = analysis.best_config
        self.model = self._create_model_from_config(best_config)
        self.model.estimate_branch_lengths(tree)
        # Copy over attributes associated with the bayesian estimator.
        self.mutation_rate = self.model.mutation_rate
        self.birth_rate = self.model.birth_rate
        self.log_likelihood = self.model.log_likelihood
        self.log_joints = self.model.log_joints
        self.posteriors = self.model.posteriors
        self.posterior_means = self.model.posterior_means

    def _trainable(self, config: Dict):
        model = self._create_model_from_config(config)
        model.estimate_branch_lengths(deepcopy(self.tree))
        tune.report(log_likelihood=model.log_likelihood)

    def _create_model_from_config(self, config):
        return IIDExponentialPosteriorMeanBLE(
            mutation_rate=config['mutation_rate'],
            birth_rate=config['birth_rate'],
            discretization_level=self.discretization_level,
            sampling_probability=config['sampling_probability'],
            enforce_parsimony=self.enforce_parsimony,
            use_cpp_implementation=self.use_cpp_implementation,
            verbose=self.verbose,
        )
