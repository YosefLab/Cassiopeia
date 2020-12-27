import abc
import copy
import cvxpy as cp
import numpy as np
from typing import List, Tuple
from .tree import Tree


class BranchLengthEstimator(abc.ABC):
    r"""
    Abstract base class for all branch length estimators.
    """
    @abc.abstractmethod
    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        Annotates the tree's nodes with their estimated age, and
        the tree's branches with their lengths.
        Operates on the tree in-place.

        Args:
            tree: The tree for which to estimate branch lengths.
        """


class IIDExponentialBLE(BranchLengthEstimator):
    r"""
    A simple branch length estimator that assumes that the characters evolve IID
    over the phylogeny with the same cutting rate.

    Maximum Parsinomy is used to impute the ancestral states first. Doing so
    leads to a convex optimization problem.
    """
    def __init__(
        self,
        minimum_edge_length: float = 0,  # TODO: minimum_branch_length?
        l2_regularization: float = 0,
        verbose: bool = False
    ):
        self.minimum_edge_length = minimum_edge_length
        self.l2_regularization = l2_regularization
        self.verbose = verbose

    def estimate_branch_lengths(self, tree: Tree) -> None:
        r"""
        Estimates branch lengths for the given tree.

        This is in fact an exponential cone program, which is a special kind of
        convex problem:
        https://docs.mosek.com/modeling-cookbook/expo.html

        Args:
            tree: The tree for which to estimate branch lengths.

        Returns:
            The log-likelihood under the model for the computed branch lengths.
        """
        # Extract parameters
        minimum_edge_length = self.minimum_edge_length
        l2_regularization = self.l2_regularization
        verbose = self.verbose

        # # Wrap the networkx DiGraph for goodies.
        # T = Tree(tree)
        T = tree

        # # # # # Create variables of the optimization problem # # # # #
        r_X_t_variables = dict([(node_id, cp.Variable(name=f'r_X_t_{node_id}'))
                                for node_id in T.nodes()])
        time_increases_constraints = [
            r_X_t_variables[parent]
            >= r_X_t_variables[child] + minimum_edge_length
            for (parent, child) in T.edges()
        ]
        leaves_have_age_0_constraints =\
            [r_X_t_variables[leaf] == 0 for leaf in T.leaves()]
        non_negative_r_X_t_constraints =\
            [r_X_t >= 0 for r_X_t in r_X_t_variables.values()]
        all_constraints =\
            time_increases_constraints + \
            leaves_have_age_0_constraints + \
            non_negative_r_X_t_constraints

        # # # # # Compute the log-likelihood # # # # #
        log_likelihood = 0

        # Because all rates are equal, the number of cuts in each node is a
        # sufficient statistic. This makes the solver WAY faster!
        for (parent, child) in T.edges():
            edge_length = r_X_t_variables[parent] - r_X_t_variables[child]
            zeros_parent = T.get_state(parent).count('0')  # TODO: '0'...
            zeros_child = T.get_state(child).count('0')  # TODO: '0'...
            new_cuts_child = zeros_parent - zeros_child
            assert(new_cuts_child >= 0)
            # Add log-lik for characters that didn't get cut
            log_likelihood += zeros_child * (-edge_length)
            # Add log-lik for characters that got cut
            log_likelihood += new_cuts_child * cp.log(1 - cp.exp(-edge_length))

        # # # # # Add regularization # # # # #

        l2_penalty = 0
        for (parent, child) in T.edges():
            for child_of_child in T.children(child):
                edge_length_above =\
                    r_X_t_variables[parent] - r_X_t_variables[child]
                edge_length_below =\
                    r_X_t_variables[child] - r_X_t_variables[child_of_child]
                l2_penalty += (edge_length_above - edge_length_below) ** 2
        l2_penalty *= l2_regularization

        # # # # # Solve the problem # # # # #

        obj = cp.Maximize(log_likelihood - l2_penalty)
        prob = cp.Problem(obj, all_constraints)

        f_star = prob.solve(solver='ECOS', verbose=verbose)

        # # # # # Populate the tree with the estimated branch lengths # # # # #

        for node in T.nodes():
            T.set_age(node, age=r_X_t_variables[node].value)

        for (parent, child) in T.edges():
            new_edge_length =\
                r_X_t_variables[parent].value - r_X_t_variables[child].value
            T.set_edge_length(
                parent,
                child,
                length=new_edge_length)

        self.log_likelihood = log_likelihood.value
        self.log_loss = f_star

    @classmethod
    def log_likelihood(self, T: Tree) -> float:
        r"""
        The log-likelihood under the model.
        """
        log_likelihood = 0.0
        for (parent, child) in T.edges():
            edge_length = T.get_age(parent) - T.get_age(child)
            zeros_parent = T.get_state(parent).count('0')  # TODO: hardcoded '0'
            zeros_child = T.get_state(child).count('0')  # TODO: hardcoded '0'
            new_cuts_child = zeros_parent - zeros_child
            assert(new_cuts_child >= 0)
            # Add log-lik for characters that didn't get cut
            log_likelihood += zeros_child * (-edge_length)
            # Add log-lik for characters that got cut
            if edge_length < 1e-8 and new_cuts_child > 0:
                return -np.inf
            log_likelihood += new_cuts_child * np.log(1 - np.exp(-edge_length))
        return log_likelihood


class IIDExponentialBLEGridSearchCV(BranchLengthEstimator):
    r"""
    Cross-validated version of IIDExponentialBLE which fits the hyperparameters
    based on character-level held-out log-likelihood.
    """
    def __init__(
        self,
        minimum_edge_lengths: Tuple[float] = (0, ),
        l2_regularizations: Tuple[float] = (0, ),
        verbose: bool = False
    ):
        self.minimum_edge_lengths = minimum_edge_lengths
        self.l2_regularizations = l2_regularizations
        self.verbose = verbose

    def estimate_branch_lengths(self, T: Tree) -> None:
        r"""
        TODO
        """
        # Extract parameters
        minimum_edge_lengths = self.minimum_edge_lengths
        l2_regularizations = self.l2_regularizations
        verbose = self.verbose

        held_out_log_likelihoods = []  # type: List[Tuple[float, List]]
        for minimum_edge_length in minimum_edge_lengths:
            for l2_regularization in l2_regularizations:
                cv_log_likelihood = self._cv_log_likelihood(
                    T=T,
                    minimum_edge_length=minimum_edge_length,
                    l2_regularization=l2_regularization)
                held_out_log_likelihoods.append(
                    (cv_log_likelihood,
                     [minimum_edge_length,
                      l2_regularization])
                )

        # Refit model on full dataset with the best hyperparameters
        held_out_log_likelihoods.sort(reverse=True)
        best_minimum_edge_length, best_l2_regularization =\
            held_out_log_likelihoods[0][1]
        if verbose:
            print(f"Refitting full model with:\n"
                  f"minimum_edge_length={best_minimum_edge_length}\n"
                  f"l2_regularization={best_l2_regularization}")
        final_model = IIDExponentialBLE(
            minimum_edge_length=best_minimum_edge_length,
            l2_regularization=best_l2_regularization
        )
        final_model.estimate_branch_lengths(T)
        self.minimum_edge_length = best_minimum_edge_length
        self.l2_regularization = best_l2_regularization
        self.log_likelihood = final_model.log_likelihood
        self.log_loss = final_model.log_loss

    def _cv_log_likelihood(
        self,
        T: Tree,
        minimum_edge_length: float,
        l2_regularization: float
    ) -> float:
        verbose = self.verbose
        if verbose:
            print(f"Cross-validating hyperparameters:"
                  f"\nminimum_edge_length={minimum_edge_length}"
                  f"\nl2_regularizations={l2_regularization}")
        n_characters = T.num_characters()
        log_likelihood_folds = np.zeros(shape=(n_characters))
        for held_out_character_idx in range(n_characters):
            T_train, T_valid =\
                self._cv_split(
                    T,
                    held_out_character_idx=held_out_character_idx
                )
            IIDExponentialBLE(
                minimum_edge_length=minimum_edge_length,
                l2_regularization=l2_regularization
            ).estimate_branch_lengths(T_train)
            T_valid.copy_branch_lengths(T_other=T_train)
            held_out_log_likelihood =\
                IIDExponentialBLE.log_likelihood(T_valid)
            log_likelihood_folds[held_out_character_idx] =\
                held_out_log_likelihood
        if verbose:
            print(f"log_likelihood_folds = {log_likelihood_folds}")
            print(f"mean log_likelihood_folds = "
                  f"{np.mean(log_likelihood_folds)}")
        return np.mean(log_likelihood_folds)

    def _cv_split(
        self,
        T: Tree,
        held_out_character_idx: int
    ) -> Tuple[Tree, Tree]:
        r"""
        Creates a training and a cross validation tree by hiding the
        character at position held_out_character_idx.
        """
        T_train = copy.deepcopy(T)
        T_valid = copy.deepcopy(T)
        for node in T.nodes():
            state = T_train.get_state(node)
            train_state =\
                state[:held_out_character_idx]\
                + state[(held_out_character_idx + 1):]
            valid_data =\
                state[held_out_character_idx]
            T_train.set_state(node, train_state)
            T_valid.set_state(node, valid_data)
        return T_train, T_valid
