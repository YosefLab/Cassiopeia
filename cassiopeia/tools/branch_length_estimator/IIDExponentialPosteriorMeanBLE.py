import multiprocessing
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from scipy import integrate
from scipy.special import binom, logsumexp

from ..tree import Tree
from .BranchLengthEstimator import BranchLengthEstimator
from . import utils


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
        enforce_parsimony: TODO
        verbose: Verbosity level. TODO

    Attributes: TODO

    """

    def __init__(
        self,
        mutation_rate: float,
        birth_rate: float,
        discretization_level: int,
        enforce_parsimony: bool = True,
    ) -> None:
        # TODO: If we use autograd, we can tune the hyperparams with gradient
        # descent?
        self.mutation_rate = mutation_rate
        # TODO: Is there some easy heuristic way to set this to a reasonable
        # value and thus avoid grid searching it / optimizing it?
        self.birth_rate = birth_rate
        self.discretization_level = discretization_level
        self.enforce_parsimony = enforce_parsimony

    def _compute_log_likelihood(self):
        tree = self.tree
        discretization_level = self.discretization_level
        log_likelihood = 0
        # TODO: Should I also add a division event when the root has multiple
        # children? (If not, the joint we are computing won't integrate to 1;
        # on the other hand, this is a constant multiplicative term that doesn't
        # affect inference.
        for child_of_root in tree.children(tree.root()):
            log_likelihood += self.down(child_of_root, discretization_level, 0)
        self.log_likelihood = log_likelihood

    def _compute_log_joint(self, v, t):
        r"""
        P(t_v = t, X, T).
        Depending on whether we are enforcing parsimony or not, we consider
        different possible number of cuts for v.
        """
        discretization_level = self.discretization_level
        tree = self.tree
        assert v in tree.internal_nodes()
        lam = self.birth_rate
        enforce_parsimony = self.enforce_parsimony
        dt = 1.0 / discretization_level
        children = tree.children(v)
        if enforce_parsimony:
            valid_num_cuts = [tree.num_cuts(v)]
        else:
            valid_num_cuts = range(tree.num_cuts(v) + 1)
        ll_for_x = []
        for x in valid_num_cuts:
            ll_for_x.append(
                sum([self.down(u, t, x) for u in children])
                + self.up(v, t, x)
                + np.log(lam * dt)
            )
        return logsumexp(ll_for_x)

    def _compute_posteriors(self):
        tree = self.tree
        discretization_level = self.discretization_level
        log_joints = {}  # log P(t_v = t, X, T)
        posteriors = {}  # P(t_v = t | X, T)
        posterior_means = {}  # E[t_v = t | X, T]
        for v in tree.internal_nodes():
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
        for node in tree.internal_nodes():
            tree.set_age(node, age=posterior_means[node])
        tree.set_age(tree.root(), age=1.0)
        for leaf in tree.leaves():
            tree.set_age(leaf, age=0.0)
        for (parent, child) in tree.edges():
            new_edge_length = tree.get_age(parent) - tree.get_age(child)
            tree.set_edge_length(parent, child, length=new_edge_length)

    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        See base class.
        """
        self.down_cache = {}  # TODO: Rename to _down_cache
        self.up_cache = {}  # TODO: Rename to _up_cache
        self.tree = tree
        self._compute_log_likelihood()
        self._compute_posteriors()
        self._populate_branch_lengths()

    def compatible_with_observed_data(self, x, observed_cuts) -> bool:
        # TODO: Make method private
        if self.enforce_parsimony:
            return x == observed_cuts
        else:
            return x <= observed_cuts

    def up(self, v, t, x) -> float:
        r"""
        TODO: Rename this _up?
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
        if not (1.0 - lam * dt - K * r * dt > 0):
            raise ValueError("Please choose a bigger discretization_level.")
        log_likelihood = 0.0
        if v == tree.root():  # Base case: we reached the root of the tree.
            # TODO: 'tree.root()' is O(n). We should have O(1) method.
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
                # TODO: 'tree.root()' is O(n). We should have O(1) method.
                p = tree.parent(v)
                if self.compatible_with_observed_data(x, tree.num_cuts(p)):
                    siblings = [u for u in tree.children(p) if u != v]
                    ll = (
                        np.log(lam * dt)
                        + self.up(p, t + 1, x)
                        + sum([self.down(u, t, x) for u in siblings])
                    )
                    if p == tree.root():  # The branch start is for free!
                        # TODO: 'tree.root()' is O(n). We should have O(1)
                        # method.
                        ll -= np.log(lam * dt)
                    log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self.up_cache[(v, t, x)] = log_likelihood
        return log_likelihood

    def down(self, v, t, x) -> float:
        r"""
        TODO: Rename this _down?
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
        if not (1.0 - lam * dt - K * r * dt > 0):
            raise ValueError("Please choose a bigger discretization_level.")
        log_likelihood = 0.0
        if t == 0:  # Base case
            if v in tree.leaves() and x == tree.num_cuts(v):
                # TODO: 'v not in tree.leaves()' is O(n). We should have O(1)
                # check.
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
            if (
                self.compatible_with_observed_data(x, tree.num_cuts(v))
                and v not in tree.leaves()
            ):
                # TODO: 'v not in tree.leaves()' is O(n). We should have O(1)
                # check.
                ll = sum(
                    [self.down(child, t - 1, x) for child in tree.children(v)]
                ) + np.log(lam * dt)
                log_likelihoods_cases.append(ll)
            log_likelihood = logsumexp(log_likelihoods_cases)
        self.down_cache[(v, t, x)] = log_likelihood
        return log_likelihood

    @classmethod
    def exact_log_full_joint(
        self, tree: Tree, mutation_rate: float, birth_rate: float
    ) -> float:
        r"""
        log P(T, X, branch_lengths), i.e. the full joint log likelihood given
        both character vectors _and_ branch lengths.
        """
        tree = deepcopy(tree)
        tree.set_edge_lengths_from_node_ages()
        ll = 0.0
        lam = birth_rate
        r = mutation_rate
        lg = np.log
        e = np.exp
        b = binom
        for (p, c) in tree.edges():
            t = tree.get_edge_length(p, c)
            # Birth process likelihood
            ll += -t * lam
            if c not in tree.leaves():
                ll += lg(lam)
            # Mutation process likelihood
            cuts = tree.number_of_mutations_along_edge(p, c)
            uncuts = tree.number_of_nonmutations_along_edge(p, c)
            ll += (
                (-t * r) * uncuts
                + lg(1 - e(-t * r)) * cuts
                + lg(b(cuts + uncuts, cuts))
            )
        return ll

    @classmethod
    def numerical_log_likelihood(
        self,
        tree: Tree,
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
            ages = args
            for node, age in list(zip(tree.internal_nodes(), ages)):
                tree.set_age(node, age)
            for (p, c) in tree.edges():
                if tree.get_age(p) <= tree.get_age(c):
                    return 0.0
            tree.set_edge_lengths_from_node_ages()
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
                [[0, 1]] * len(tree.internal_nodes()),
                opts={"epsrel": epsrel},
            )[0]
        )
        assert not np.isnan(res)
        return res

    @classmethod
    def numerical_log_joint(
        self,
        tree: Tree,
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
        other_nodes = [n for n in tree.internal_nodes() if n != node]

        tree = deepcopy(tree)

        def f(*args):
            ages = args
            for other_node, age in list(zip(other_nodes, ages)):
                tree.set_age(other_node, age)
            for (p, c) in tree.edges():
                if tree.get_age(p) <= tree.get_age(c):
                    return 0.0
            tree.set_edge_lengths_from_node_ages()
            return np.exp(
                IIDExponentialPosteriorMeanBLE.exact_log_full_joint(
                    tree=tree,
                    mutation_rate=mutation_rate,
                    birth_rate=birth_rate,
                )
            )

        for i in range(discretization_level + 1):
            node_age = i / discretization_level
            tree.set_age(node, node_age)
            tree.set_edge_lengths_from_node_ages()
            if len(other_nodes) == 0:
                # There is nothing to integrate over.
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
                            [[0, 1]] * (len(tree.internal_nodes()) - 1),
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
        tree: Tree,
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
        enforce_parsimony: bool = True,
        processes: int = 6,
        verbose: bool = False,
    ):
        self.mutation_rates = mutation_rates
        self.birth_rates = birth_rates
        self.discretization_level = discretization_level
        self.enforce_parsimony = enforce_parsimony
        self.processes = processes
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        See base class.
        """
        mutation_rates = self.mutation_rates
        birth_rates = self.birth_rates
        discretization_level = self.discretization_level
        enforce_parsimony = self.enforce_parsimony
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
                        discretization_level=discretization_level,
                        enforce_parsimony=enforce_parsimony,
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
            discretization_level=discretization_level,
            enforce_parsimony=enforce_parsimony,
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
        self,
        figure_file: Optional[str] = None,
        show_plot: bool = True
    ):
        utils.plot_grid(
            grid=self.grid,
            yticklabels=self.mutation_rates,
            xticklabels=self.birth_rates,
            ylabel=r"Mutation Rate ($r$)",
            xlabel=r"Birth Rate ($\lambda$)",
            figure_file=figure_file,
            show_plot=show_plot,
        )
