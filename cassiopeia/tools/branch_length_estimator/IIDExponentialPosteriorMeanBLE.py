from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from .BranchLengthEstimator import BranchLengthEstimator
from ..tree import Tree


class IIDExponentialPosteriorMeanBLE(BranchLengthEstimator):
    r"""
    Same model as IIDExponentialBLE but computes the posterior mean instead
    of the MLE. The phylogeny model is chosen to be a birth process.

    This estimator requires that the ancestral states are provided.
    TODO: Allow for two versions: one where the number of mutations of each
    node must match exactly, and one where it must be upper bounded by the
    number of mutations seen. (I believe the latter should ameliorate
    subtree collapse further.)

    TODO: Use numpy autograd to do optimize the hyperparams? (Empirical Bayes)

    We compute the posterior means using a forward-backward-style algorithm
    (DP on a tree).

    Args:
        mutation_rate: TODO
        birth_rate: TODO
        discretization_level: TODO
        verbose: Verbosity level. TODO

    Attributes: TODO

    """

    def __init__(
        self, mutation_rate: float, birth_rate: float, discretization_level: int
    ) -> None:
        # TODO: If we use autograd, we can tune the hyperparams with gradient
        # descent.
        self.mutation_rate = mutation_rate
        # TODO: Is there some easy heuristic way to set this to a reasonable
        # value and thus avoid grid searching it / optimizing it?
        self.birth_rate = birth_rate
        self.discretization_level = discretization_level

    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        See base class.
        """
        discretization_level = self.discretization_level
        self.down_cache = {}  # TODO: Rename to _down_cache
        self.up_cache = {}  # TODO: Rename to _up_cache
        self.tree = tree
        log_likelihood = 0
        # TODO: Should I also add a division event when the root has multiple
        # children?
        for child_of_root in tree.children(tree.root()):
            log_likelihood += self.down(child_of_root, discretization_level, 0)
        self.log_likelihood = log_likelihood
        # # # # # Compute Posteriors # # # # #
        posteriors = {}
        log_posteriors = {}
        posterior_means = {}
        for v in tree.internal_nodes():
            # Compute the posterior for this node
            posterior = np.zeros(shape=(discretization_level + 1,))
            for t in range(discretization_level + 1):
                posterior[t] = self.down(v, t, tree.num_cuts(v)) + self.up(
                    v, t, tree.num_cuts(v)
                )
            posterior -= np.max(posterior)
            log_posteriors[v] = posterior.copy()
            posterior = np.exp(posterior)
            posterior /= np.sum(posterior)
            posteriors[v] = posterior
            posterior_means[v] = (
                posterior * np.array(range(discretization_level + 1))
            ).sum() / discretization_level
        self.posteriors = posteriors
        self.log_posteriors = log_posteriors
        self.posterior_means = posterior_means
        # # # # # Populate the tree with the estimated branch lengths # # # # #
        for node in tree.internal_nodes():
            tree.set_age(node, age=posterior_means[node])
        tree.set_age(tree.root(), age=1.0)
        for leaf in tree.leaves():
            tree.set_age(leaf, age=0.0)

        for (parent, child) in tree.edges():
            new_edge_length = tree.get_age(parent) - tree.get_age(child)
            tree.set_edge_length(parent, child, length=new_edge_length)

    def up(self, v, t, x) -> float:
        r"""
        TODO: Rename this _up.
        log P(X_up(b(v)), T_up(b(v)), t \in t_b(v), X_b(v)(t) = x)
        """
        if (v, t, x) in self.up_cache:
            # TODO: Use a decorator instead of a hand-made cache
            return self.up_cache[(v, t, x)]
        # Pull out params
        r = self.mutation_rate
        lam = self.birth_rate
        dt = 1.0 / self.discretization_level
        K = self.tree.num_characters()
        tree = self.tree
        discretization_level = self.discretization_level
        assert 0 <= t <= self.discretization_level
        assert 0 <= x <= K
        log_likelihood = 0.0
        if v == tree.root():  # Base case: we reached the root of the tree.
            if t == discretization_level and x == tree.num_cuts(v):
                log_likelihood = 0.0
            else:
                log_likelihood = -np.inf
        elif t == discretization_level:
            # Base case: we reached the start of the process, but we're not yet
            # at the root.
            assert v != tree.root()
            log_likelihood = -np.inf
        else:  # Recursion.
            log_likelihoods_cases = []
            # Case 1: Nothing happened
            log_likelihoods_cases.append(
                np.log(1.0 - lam * dt - (K - x) * r * dt) + self.up(v, t + 1, x)
            )
            # Case 2: Mutation happened
            if x - 1 >= 0:
                log_likelihoods_cases.append(
                    np.log((K - (x - 1)) * r * dt) + self.up(v, t + 1, x - 1)
                )
            # Case 3: A cell division happened
            if v != tree.root():
                p = tree.parent(v)
                if x == tree.num_cuts(p):
                    siblings = [u for u in tree.children(p) if u != v]
                    ll = (
                        np.log(lam * dt)
                        + self.up(p, t + 1, x)
                        + sum([self.down(u, t, x) for u in siblings])
                    )
                    if p == tree.root():  # The branch start is for free!
                        ll -= np.log(lam * dt)
                    log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self.up_cache[(v, t, x)] = log_likelihood
        return log_likelihood

    def down(self, v, t, x) -> float:
        r"""
        TODO: Rename this _down.
        log P(X_down(v), T_down(v) | t_v = t, X_v = x)
        """
        if (v, t, x) in self.down_cache:
            # TODO: Use a decorator instead of a hand-made cache
            return self.down_cache[(v, t, x)]
        # Pull out params
        r = self.mutation_rate
        lam = self.birth_rate
        dt = 1.0 / self.discretization_level
        K = self.tree.num_characters()
        tree = self.tree
        assert v != tree.root()
        assert 0 <= t <= self.discretization_level
        assert 0 <= x <= K
        log_likelihood = 0.0
        if t == 0:  # Base case
            if v in tree.leaves() and x == tree.num_cuts(v):
                log_likelihood = 0.0
            else:
                log_likelihood = -np.inf
        else:  # Recursion.
            log_likelihoods_cases = []
            # Case 1: Nothing happens
            log_likelihoods_cases.append(
                np.log(1.0 - lam * dt - (K - x) * r * dt)
                + self.down(v, t - 1, x)
            )
            # Case 2: One character mutates.
            if x + 1 <= K:
                log_likelihoods_cases.append(
                    np.log((K - x) * r * dt) + self.down(v, t - 1, x + 1)
                )
            # Case 3: Cell divides
            # The number of cuts at this state must match the ground truth.
            # TODO: Allow for weak match at internal nodes and exact match at
            # leaves.
            if x == tree.num_cuts(v) and v not in tree.leaves():
                ll = sum(
                    [self.down(child, t - 1, x) for child in tree.children(v)]
                ) + np.log(lam * dt)
                log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self.down_cache[(v, t, x)] = log_likelihood
        return log_likelihood


class IIDExponentialPosteriorMeanBLEGridSearchCV(BranchLengthEstimator):
    r"""
    Like IIDExponentialPosteriorMeanBLE but with automatic tuning of
    hyperparameters.

    This class fits the hyperparameters of IIDExponentialPosteriorMeanBLE based
    on data log-likelihood. I.e. is performs empirical Bayes.

    Args:
        mutation_rates: TODO
        birth_rate: TODO
        discretization_level: TODO
        verbose: Verbosity level. TODO
    """

    def __init__(
        self,
        mutation_rates: Tuple[float] = (0,),
        birth_rates: Tuple[float] = (0,),
        discretization_level: int = 1000,
        verbose: bool = False,
    ):
        self.mutation_rates = mutation_rates
        self.birth_rates = birth_rates
        self.discretization_level = discretization_level
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        See base class.
        """
        mutation_rates = self.mutation_rates
        birth_rates = self.birth_rates
        discretization_level = self.discretization_level
        verbose = self.verbose
        lls = []
        grid = np.zeros(shape=(len(mutation_rates), len(birth_rates)))
        for i, mutation_rate in enumerate(mutation_rates):
            for j, birth_rate in enumerate(birth_rates):
                if self.verbose:
                    print(
                        f"Fitting model with:\n"
                        f"best_mutation_rate={mutation_rate}\n"
                        f"best_birth_rate={birth_rate}"
                    )
                model = IIDExponentialPosteriorMeanBLE(
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                    discretization_level=discretization_level,
                )
                model.estimate_branch_lengths(tree)
                ll = model.log_likelihood
                lls.append((ll, (mutation_rate, birth_rate)))
                grid[i, j] = ll
        lls.sort(reverse=True)
        (best_mutation_rate, best_birth_rate,) = lls[
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
            discretization_level=discretization_level,
        )
        final_model.estimate_branch_lengths(tree)
        self.mutation_rate = best_mutation_rate
        self.birth_rate = best_birth_rate
        self.log_likelihood = final_model.log_likelihood
        self.posteriors = final_model.posteriors
        self.log_posteriors = final_model.log_posteriors
        self.posterior_means = final_model.posterior_means
        self.grid = grid
