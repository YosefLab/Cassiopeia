"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
from collections import defaultdict
import cvxpy as cp
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import IIDExponentialMLEError

from ._branch_length_estimator import BranchLengthEstimator


class IIDExponentialMLE(BranchLengthEstimator):
    """
    MLE under a model of IID memoryless CRISPR/Cas9 mutations.

    In more detail, this model assumes that CRISPR/Cas9 mutates each site
    independently and identically, with an exponential waiting time. The
    tree is assumed to have depth exactly 1, and the user can provide a
    minimum branch length. The MLE under this set of assumptions can be
    solved with a special kind of convex optimization problem known as an
    exponential cone program, which can be readily solved with off-the-shelf
    (open source) solvers.

    This estimator requires that the ancestral characters be provided (these
    can be imputed with CassiopeiaTree's reconstruct_ancestral_characters
    method if they are not known, which is usually the case for real data).

    The estimated mutation rate under will be stored as an attribute called
    `mutation_rate`. The log-likelihood will be stored in an attribute
    called `log_likelihood`.

    Missing states are treated as missing at random by the model.

    Args:
        minimum_branch_length: Estimated branch lengths will be constrained to
            have length at least this value. By default it is set to 0.01,
            since the MLE tends to collapse mutationless edges to length 0.
        l1_regularization: Consecutive branches will be regularized to have
            similar length via an L1 penalty whose weight is given by
            l1_regularization.
        l2_regularization: Consecutive branches will be regularized to have
            similar length via an L2 penalty whose weight is given by
            l2_regularization.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
        verbose: Verbosity level.

    Attributes:
        mutation_rate: The estimated CRISPR/Cas9 mutation rate, assuming that
            the tree has depth exactly 1.
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        minimum_branch_length: The minimum branch length.
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        l1_regularization: float = 0.0,
        l2_regularization: float = 0.0,
        pseudo_mutations_per_edge: float = 0.0,
        pseudo_non_mutations_per_edge: float = 0.0,
        verbose: bool = False,
        solver: str = "SCS",
    ):
        allowed_solvers = ["ECOS", "SCS", "MOSEK"]
        if solver not in allowed_solvers:
            raise ValueError(
                f"Solver {solver} not allowed. "
                f"Allowed solvers: {allowed_solvers}"
            )  # pragma: no cover
        self._minimum_branch_length = minimum_branch_length
        self._l1_regularization = l1_regularization
        self._l2_regularization = l2_regularization
        self._pseudo_mutations_per_edge = pseudo_mutations_per_edge
        self._pseudo_non_mutations_per_edge = pseudo_non_mutations_per_edge
        self._verbose = verbose
        self._solver = solver
        self._mutation_rate = None
        self._log_likelihood = None

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        MLE under a model of IID memoryless CRISPR/Cas9 mutations.

        The only caveat is that this method raises an IIDExponentialMLEError
        if the underlying convex optimization solver fails, or a
        ValueError if the character matrix is degenerate (fully mutated,
        or fully unmutated).

        Raises:
            IIDExponentialMLEError
            ValueError
        """
        # Extract parameters
        minimum_branch_length = self._minimum_branch_length
        l1_regularization = self._l1_regularization
        l2_regularization = self._l2_regularization
        pseudo_mutations_per_edge = self._pseudo_mutations_per_edge
        pseudo_non_mutations_per_edge = self._pseudo_non_mutations_per_edge
        solver = self._solver
        verbose = self._verbose

        # # # # # Check that the character has at least one mutation # # # # #
        if (tree.character_matrix == 0).all().all():
            raise ValueError(
                "The character matrix has no mutations. Please check your data."
            )

        # # # # # Check that the character is not saturated # # # # #
        if (tree.character_matrix != 0).all().all():
            raise ValueError(
                "The character matrix is fully mutated. The MLE does not "
                "exist. Please check your data."
            )

        # # # # # Check that the minimum_branch_length makes sense # # # # #
        if tree.get_edge_depth() * minimum_branch_length >= 1.0:
            raise ValueError(
                "The minimum_branch_length is too large. Please reduce it."
            )

        # # # # # Create variables of the optimization problem # # # # #
        r_X_t_variables = dict(
            [
                (node_id, cp.Variable(name=f"r_X_t_{node_id}"))
                for node_id in tree.nodes
            ]
        )

        # # # # # Create constraints of the optimization problem # # # # #
        a_leaf = tree.leaves[0]
        root = tree.root
        root_has_time_0_constraint = [r_X_t_variables[root] == 0]
        minimum_branch_length_constraints = [
            r_X_t_variables[child]
            >= r_X_t_variables[parent]
            + minimum_branch_length * r_X_t_variables[a_leaf]
            for (parent, child) in tree.edges
        ]
        ultrametric_constraints = [
            r_X_t_variables[leaf] == r_X_t_variables[a_leaf]
            for leaf in tree.leaves
            if leaf != a_leaf
        ]
        all_constraints = (
            root_has_time_0_constraint
            + minimum_branch_length_constraints
            + ultrametric_constraints
        )

        # # # # # Deal with 'long-edge' mutations # # # # #
        long_edge_mutations = defaultdict(float)
        # We pre-compute all states since we will need repeated access
        character_states_dict = {
            node: tree.get_character_states(node)
            for node in tree.nodes
        }
        k = tree.character_matrix.shape[1]
        for node in tree.nodes:
            if tree.is_root(node):
                continue
            parent = tree.parent(node)
            character_states = character_states_dict[node]
            parent_states = character_states_dict[parent]
            for i in range(k):
                if character_states[i] > 0 and parent_states[i] == tree.missing_state_indicator:
                    # Need to go up the tree and determine if we have a long
                    # edge mutation.
                    u = parent
                    while character_states_dict[u][i] == tree.missing_state_indicator:
                        u = tree.parent(u)
                    if character_states_dict[u][i] == 0:
                        # We have identified a 'long-edge' mutation
                        long_edge_mutations[(u, node)] += 1
                    else:
                        if character_states_dict[u][i] != character_states[i]:
                            raise Exception(
                                "Ancestral state reconstruction seems invalid: "
                                f" character {character_states[i]} descends "
                                f"from {character_states_dict[u][i]}."
                            )
        del character_states_dict

        # # # # # Compute the log-likelihood for edges # # # # #
        log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            num_unmutated = len(
                tree.get_unmutated_characters_along_edge(parent, child)
            ) + pseudo_non_mutations_per_edge
            num_mutated = len(
                tree.get_mutations_along_edge(
                    parent, child, treat_missing_as_mutations=False
                )
            ) + pseudo_mutations_per_edge
            log_likelihood += num_unmutated * (-edge_length)
            log_likelihood += num_mutated * cp.log(
                1 - cp.exp(-edge_length - 1e-5)  # We add eps for stability.
            )

        # # # # # Add in log-likelihood of long-edge mutations # # # # #
        for ((parent, child), num_mutated) in long_edge_mutations.items():
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            log_likelihood += num_mutated * cp.log(
                1 - cp.exp(-edge_length - 1e-5)  # We add eps for stability.
            )

        # # # # # Normalize log_likelihood by the number of sites # # # # #
        # This is just to keep the log-likelihood on a similar scale
        # regardless of the number of characters.
        log_likelihood /= tree.character_matrix.shape[1]

        # # # # # Add L1 regularization # # # # #

        l1_penalty = 0
        if l1_regularization > 0:
            for (parent, child) in tree.edges:
                for child_of_child in tree.children(child):
                    edge_length_above = (
                        r_X_t_variables[child] - r_X_t_variables[parent]
                    )
                    edge_length_below = (
                        r_X_t_variables[child_of_child] - r_X_t_variables[child]
                    )
                    l1_penalty += cp.abs(edge_length_above - edge_length_below)
            l1_penalty *= l1_regularization

        # # # # # Add L2 regularization # # # # #

        l2_penalty = 0
        if l2_regularization > 0:
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

        obj = cp.Maximize(log_likelihood - l1_penalty - l2_penalty)
        prob = cp.Problem(obj, all_constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except cp.SolverError:  # pragma: no cover
            raise IIDExponentialMLEError("Third-party solver failed")

        # # # # # Extract the mutation rate # # # # #
        self._mutation_rate = float(r_X_t_variables[a_leaf].value)
        if self._mutation_rate < 1e-8 or self._mutation_rate > 15.0:
            raise IIDExponentialMLEError(
                "The solver failed when it shouldn't have."
            )

        # # # # # Extract the log-likelihood # # # # #
        # Need to re-scale by the number of characters
        log_likelihood = float(log_likelihood.value) \
            * tree.character_matrix.shape[1]
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
        self._log_likelihood = log_likelihood

        # # # # # Populate the tree with the estimated branch lengths # # # # #
        times = {
            node: float(r_X_t_variables[node].value) / self._mutation_rate
            for node in tree.nodes
        }
        # Make sure that the root has time 0 (avoid epsilons)
        times[tree.root] = 0.0
        # We smooth out epsilons that might make a parent's time greater
        # than its child (which can happen if minimum_branch_length=0)
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        # Make sure all leaves have the same time.
        tree_depth = max([times[leaf] for leaf in tree.leaves])
        for leaf in tree.leaves:
            times[leaf] = tree_depth
        tree.set_times(times)

    @property
    def log_likelihood(self):
        """
        The log-likelihood of the training data under the estimated model.
        """
        return self._log_likelihood

    @property
    def mutation_rate(self):
        """
        The estimated CRISPR/Cas9 mutation rate under the given model.
        """
        return self._mutation_rate

    @property
    def minimum_branch_length(self):
        """
        The minimum_branch_length.
        """
        return self._minimum_branch_length
