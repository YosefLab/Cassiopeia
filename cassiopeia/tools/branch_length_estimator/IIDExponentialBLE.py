import multiprocessing
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

from cassiopeia.data import CassiopeiaTree
from .BranchLengthEstimator import BranchLengthEstimator
from . import utils


class IIDExponentialBLE(BranchLengthEstimator):
    r"""
    A simple branch length estimator that assumes that the characters evolve IID
    over the phylogeny with the same cutting rate.

    This estimator requires that the ancestral states are provided.

    The optimization problem is a special kind of convex program called an
    exponential cone program:
    https://docs.mosek.com/modeling-cookbook/expo.html
    Because it is a convex program, it can be readily solved.

    Args:
        minimum_branch_length: Estimated branch lengths will be constrained to
            have at least length THIS MULTIPLE OF THE TREE HEIGHT. If this is
            greater than 1.0 / [height of the tree] (where the height
            is measured in terms of the greatest number of edges of any lineage)
            then all edges will have length 0, so be careful!
        l2_regularization: Consecutive branches will be regularized to have
            similar length via an L2 penalty whose weight is given by
            l2_regularization.
        treat_missing_states_as_mutations: If True, missing states will be treated as
            their own CRISPR/Cas9 mutations.
        verbose: Verbosity level.

    Attributes:
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        log_loss: The log-loss of the training data under the estimated model.
            This is the log likelihood plus the regularization terms.
    """

    def __init__(
        self,
        minimum_branch_length: float = 0,
        l2_regularization: float = 0,
        treat_missing_states_as_mutations: bool = True,
        verbose: bool = False,
    ):
        self.minimum_branch_length = minimum_branch_length
        self.l2_regularization = l2_regularization
        self.treat_missing_states_as_mutations = treat_missing_states_as_mutations
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class. The only caveat is that this method raises if it fails
        to solve the underlying optimization problem for any reason.

        Raises:
            cp.error.SolverError
        """
        # Extract parameters
        minimum_branch_length = self.minimum_branch_length
        l2_regularization = self.l2_regularization
        treat_missing_states_as_mutations = self.treat_missing_states_as_mutations
        verbose = self.verbose

        # # # # # Create variables of the optimization problem # # # # #
        r_X_t_variables = dict(
            [
                (node_id, cp.Variable(name=f"r_X_t_{node_id}"))
                for node_id in tree.nodes
            ]
        )
        a_leaf = tree.leaves[0]
        root = tree.root
        root_has_time_0_constraint = [r_X_t_variables[root] == 0]
        time_increases_constraints = [
            r_X_t_variables[child]
            >= r_X_t_variables[parent]
            + minimum_branch_length * r_X_t_variables[a_leaf]
            for (parent, child) in tree.edges
        ]
        leaves_have_same_time_constraints = [
            r_X_t_variables[leaf] == r_X_t_variables[a_leaf]
            for leaf in tree.leaves
            if leaf != a_leaf
        ]
        non_negative_r_X_t_constraints = [
            r_X_t >= 0 for r_X_t in r_X_t_variables.values()
        ]
        all_constraints = (
            root_has_time_0_constraint
            + time_increases_constraints
            + leaves_have_same_time_constraints
            + non_negative_r_X_t_constraints
        )

        # # # # # Compute the log-likelihood # # # # #
        log_likelihood = 0

        # Because all rates are equal, the number of cuts in each node is a
        # sufficient statistic. This makes the solver WAY faster!
        for (parent, child) in tree.edges:
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            num_uncuts = 0
            num_cuts = 0
            num_missing = 0
            for parent_state, child_state in zip(parent_states, child_states):
                # We only care about uncut states.
                if parent_state == 0:
                    if child_state == 0:
                        num_uncuts += 1
                    elif child_state == tree.missing_state_indicator:
                        num_missing += 1
                    else:
                        num_cuts += 1
            if treat_missing_states_as_mutations:
                # TODO: Test this functionality!
                num_cuts += num_missing
                num_missing = 0
            log_likelihood += num_uncuts * (-edge_length)
            # Add log-lik for characters that got cut
            log_likelihood += num_cuts * cp.log(
                1 - cp.exp(-edge_length - 1e-8)
            )

        # # # # # Add regularization # # # # #

        l2_penalty = 0
        for (parent, child) in tree.edges:
            for child_of_child in tree.children(child):
                edge_length_above = (
                    r_X_t_variables[child] - r_X_t_variables[parent]
                )
                edge_length_below = (
                    r_X_t_variables[child_of_child] - r_X_t_variables[child]
                )
                l2_penalty += (edge_length_above - edge_length_below) ** 2
        l2_penalty *= l2_regularization

        # # # # # Solve the problem # # # # #

        obj = cp.Maximize(log_likelihood - l2_penalty)
        prob = cp.Problem(obj, all_constraints)

        f_star = prob.solve(solver="ECOS", verbose=verbose)

        # # # # # Populate the tree with the estimated branch lengths # # # # #

        times = {node: r_X_t_variables[node].value for node in tree.nodes}
        # We smooth out epsilons that might make a parent's time greater
        # than its child
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        tree.set_times(times)

        log_likelihood = log_likelihood.value
        log_loss = f_star
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
        if np.isnan(log_loss):
            log_loss = -np.inf
        self.log_likelihood = log_likelihood
        self.log_loss = log_loss
        self.mutation_rate = tree.get_depth()

    @classmethod
    def log_likelihood(self, tree: CassiopeiaTree) -> float:
        r"""
        The log-likelihood of the given tree under the model.
        """
        log_likelihood = 0.0
        for (parent, child) in tree.edges:
            edge_length = tree.get_branch_length(parent, child)
            n_mutated = tree.get_number_of_mutations_along_edge(parent, child)
            n_nonmutated = tree.get_number_of_unmutated_characters_in_node(
                child
            )
            assert n_mutated >= 0 and n_nonmutated >= 0
            # Add log-lik for characters that didn't get cut
            log_likelihood += n_nonmutated * (-edge_length)
            # Add log-lik for characters that got cut
            if n_mutated > 0:
                if edge_length < 1e-8:
                    return -np.inf
                log_likelihood += n_mutated * np.log(1 - np.exp(-edge_length))
        assert not np.isnan(log_likelihood)
        return log_likelihood


class IIDExponentialBLEGridSearchCV(BranchLengthEstimator):
    r"""
    Like IIDExponentialBLE but with automatic tuning of hyperparameters.

    This class fits the hyperparameters of IIDExponentialBLE based on
    character-level held-out log-likelihood. It leaves out one character at a
    time, fitting the data on all the remaining characters. Thus, the number
    of models trained by this class is #characters * grid size.

    Args:
        minimum_branch_lengths: The grid of minimum_branch_length to use.
        l2_regularizations: The grid of l2_regularization to use.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        minimum_branch_lengths: Tuple[float] = (0,),
        l2_regularizations: Tuple[float] = (0,),
        processes: int = 6,
        verbose: bool = False,
    ):
        self.minimum_branch_lengths = minimum_branch_lengths
        self.l2_regularizations = l2_regularizations
        self.processes = processes
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class. The only caveat is that this method raises if it fails
        to solve the underlying optimization problem for any reason.

        Raises:
            cp.error.SolverError
        """
        # Extract parameters
        minimum_branch_lengths = self.minimum_branch_lengths
        l2_regularizations = self.l2_regularizations
        verbose = self.verbose

        held_out_log_likelihoods = []  # type: List[Tuple[float, List]]
        grid = np.zeros(
            shape=(len(minimum_branch_lengths), len(l2_regularizations))
        )
        for i, minimum_branch_length in enumerate(minimum_branch_lengths):
            for j, l2_regularization in enumerate(l2_regularizations):
                cv_log_likelihood = self._cv_log_likelihood(
                    tree=tree,
                    minimum_branch_length=minimum_branch_length,
                    l2_regularization=l2_regularization,
                )
                held_out_log_likelihoods.append(
                    (
                        cv_log_likelihood,
                        [minimum_branch_length, l2_regularization],
                    )
                )
                grid[i, j] = cv_log_likelihood

        # Refit model on full dataset with the best hyperparameters
        held_out_log_likelihoods.sort(reverse=True)
        (
            best_minimum_branch_length,
            best_l2_regularization,
        ) = held_out_log_likelihoods[0][1]
        if verbose:
            print(
                f"Refitting full model with:\n"
                f"minimum_branch_length={best_minimum_branch_length}\n"
                f"l2_regularization={best_l2_regularization}"
            )
        final_model = IIDExponentialBLE(
            minimum_branch_length=best_minimum_branch_length,
            l2_regularization=best_l2_regularization,
        )
        final_model.estimate_branch_lengths(tree)
        self.minimum_branch_length = best_minimum_branch_length
        self.l2_regularization = best_l2_regularization
        self.log_likelihood = final_model.log_likelihood
        self.log_loss = final_model.log_loss
        self.grid = grid

    def _cv_log_likelihood(
        self,
        tree: CassiopeiaTree,
        minimum_branch_length: float,
        l2_regularization: float,
    ) -> float:
        r"""
        Given the tree and the parameters of the model, returns the
        cross-validated log-likelihood of the model. This is done by holding out
        one character at a time, fitting the model on the remaining characters,
        and evaluating the log-likelihood on the held-out character. As a
        consequence, #character models are fit by this method. The mean held-out
        log-likelihood over the #character folds is returned.
        """
        verbose = self.verbose
        processes = self.processes
        if verbose:
            print(
                f"Cross-validating hyperparameters:"
                f"\nminimum_branch_length={minimum_branch_length}"
                f"\nl2_regularizations={l2_regularization}"
            )
        n_characters = tree.n_character
        params = []
        for held_out_character_idx in range(n_characters):
            train_tree, valid_tree = self._cv_split(
                tree=tree, held_out_character_idx=held_out_character_idx
            )
            model = IIDExponentialBLE(
                minimum_branch_length=minimum_branch_length,
                l2_regularization=l2_regularization,
            )
            params.append((model, train_tree, valid_tree))
        with multiprocessing.Pool(processes=processes) as pool:
            map_fn = pool.map if processes > 1 else map
            log_likelihood_folds = list(map_fn(_fit_model, params))
        if verbose:
            print(f"log_likelihood_folds = {log_likelihood_folds}")
        return np.mean(np.array(log_likelihood_folds))

    def _cv_split(
        self, tree: CassiopeiaTree, held_out_character_idx: int
    ) -> Tuple[CassiopeiaTree, CassiopeiaTree]:
        r"""
        Creates a training and a cross validation tree by hiding the
        character at position held_out_character_idx.
        """
        tree_topology = tree.get_tree_topology()
        train_states = {}
        valid_states = {}
        for node in tree.nodes:
            state = tree.get_character_states(node)
            train_state = (
                state[:held_out_character_idx]
                + state[(held_out_character_idx + 1) :]
            )
            valid_state = [state[held_out_character_idx]]
            train_states[node] = train_state
            valid_states[node] = valid_state
        train_tree = CassiopeiaTree(tree=tree_topology)
        valid_tree = CassiopeiaTree(tree=tree_topology)
        train_tree.set_all_character_states(train_states)
        valid_tree.set_all_character_states(valid_states)
        return train_tree, valid_tree

    def plot_grid(
        self, figure_file: Optional[str] = None, show_plot: bool = True
    ):
        utils.plot_grid(
            grid=self.grid,
            yticklabels=self.minimum_branch_lengths,
            xticklabels=self.l2_regularizations,
            ylabel=r"Minimum Branch Length ($\epsilon$)",
            xlabel=r"l2 Regularization ($\lambda$)",
            figure_file=figure_file,
            show_plot=show_plot,
        )


def _fit_model(args):
    r"""
    This is used by IIDExponentialBLEGridSearchCV to
    parallelize the CV folds. It must be defined here (at the top level of
    the module) for multiprocessing to be able to pickle it. (This is why
    coverage misses it)
    """
    model, train_tree, valid_tree = args
    assert valid_tree.n_character == 1
    try:
        model.estimate_branch_lengths(train_tree)
        valid_tree.set_times(train_tree.get_times())
        held_out_log_likelihood = IIDExponentialBLE.log_likelihood(valid_tree)
    except cp.error.SolverError:
        held_out_log_likelihood = -np.inf
    return held_out_log_likelihood
