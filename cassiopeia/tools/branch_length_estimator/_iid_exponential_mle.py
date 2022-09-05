"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
    minimum branch length. Pseudocounts in the form of fictitious mutations and
    non-mutations can be added to regularize the MLE. Also, the relative depth
    of each leaf can be specified to relax the ultrametric constraint. The MLE
    under this set of assumptions can be solved with a special kind of convex
    optimization problem known as an exponential cone program, which can be
    readily solved with off-the-shelf (open source) solvers.

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
        pseudo_mutations_per_edge: Regularization whereby we add this number of
            fictitious mutations to each edge in the tree.
        pseudo_non_mutations_per_edge: Regularization whereby we add this number
            of fictitious non-mutations to each edge in the tree.
        relative_leaf_depth: If provided, the relative depth of each leaf in the
            tree. This allows relaxing the ultrametric assumption to deal with
            the case where the tree is not ultrametric but the relative leaf
            depths are known.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
        verbose: Verbosity level.

    Attributes:
        mutation_rate: The estimated CRISPR/Cas9 mutation rate, assuming that
            the tree has depth exactly 1.
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        minimum_branch_length: The minimum branch length.
        pseudo_mutations_per_edge: The number of fictitious mutations added to
            each edge to regularize the MLE.
        pseudo_non_mutations_per_edge: The number of fictitious non-mutations
            added to each edge to regularize the MLE.
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        pseudo_mutations_per_edge: float = 0.0,
        pseudo_non_mutations_per_edge: float = 0.0,
        relative_leaf_depth: Optional[List[Tuple[str, float]]] = None,
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
        self._pseudo_mutations_per_edge = pseudo_mutations_per_edge
        self._pseudo_non_mutations_per_edge = pseudo_non_mutations_per_edge
        self._relative_leaf_depth = relative_leaf_depth
        self._verbose = verbose
        self._solver = solver
        self._mutation_rate = None
        self._penalized_log_likelihood = None
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
        pseudo_mutations_per_edge = self._pseudo_mutations_per_edge
        pseudo_non_mutations_per_edge = self._pseudo_non_mutations_per_edge
        relative_leaf_depth = self._relative_leaf_depth
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
        if relative_leaf_depth is None:
            relative_leaf_depth = [(leaf, 1.0) for leaf in tree.leaves]
        relative_leaf_depth = dict(relative_leaf_depth)
        ultrametric_constraints = [
            r_X_t_variables[leaf]
            == r_X_t_variables[a_leaf]
            * relative_leaf_depth[leaf]
            / relative_leaf_depth[a_leaf]
            for leaf in tree.leaves
            if leaf != a_leaf
        ]
        all_constraints = (
            root_has_time_0_constraint
            + minimum_branch_length_constraints
            + ultrametric_constraints
        )

        # # # # # Compute the log-likelihood for edges # # # # #
        penalized_log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            num_unmutated = (
                len(tree.get_unmutated_characters_along_edge(parent, child))
                + pseudo_non_mutations_per_edge
            )
            num_mutated = (
                len(
                    tree.get_mutations_along_edge(
                        parent, child, treat_missing_as_mutations=False
                    )
                )
                + pseudo_mutations_per_edge
            )
            if num_unmutated > 0:
                penalized_log_likelihood += num_unmutated * (-edge_length)
            if num_mutated > 0:
                penalized_log_likelihood += num_mutated * cp.log(
                    1 - cp.exp(-edge_length - 1e-5)  # We add eps for stability.
                )

        # # # # # Add in log-likelihood of long-edge mutations # # # # #
        long_edge_mutations = self._get_long_edge_mutations(tree)
        for ((parent, child), num_mutated) in long_edge_mutations.items():
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            penalized_log_likelihood += num_mutated * cp.log(
                1 - cp.exp(-edge_length - 1e-5)  # We add eps for stability.
            )

        # # # Normalize penalized_log_likelihood by the number of sites # # #
        # This is just to keep the log-likelihood on a similar scale
        # regardless of the number of characters.
        penalized_log_likelihood /= tree.character_matrix.shape[1]

        # # # # # Solve the problem # # # # #

        obj = cp.Maximize(penalized_log_likelihood)
        prob = cp.Problem(obj, all_constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except cp.SolverError:  # pragma: no cover
            raise IIDExponentialMLEError("Third-party solver failed")

        # # # # # Extract the mutation rate # # # # #
        max_r_X_t_value = max(
            [float(r_X_t_variables[leaf].value) for leaf in tree.leaves]
        )
        self._mutation_rate = max_r_X_t_value
        if self._mutation_rate < 1e-8 or self._mutation_rate > 15.0:
            raise IIDExponentialMLEError(
                "The solver failed when it shouldn't have."
            )

        # # # # # Extract the log-likelihood # # # # #
        # Need to re-scale by the number of characters
        penalized_log_likelihood = (
            float(penalized_log_likelihood.value)
            * tree.character_matrix.shape[1]
        )
        if np.isnan(penalized_log_likelihood):
            penalized_log_likelihood = -np.inf
        self._penalized_log_likelihood = penalized_log_likelihood

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
        self._log_likelihood = self.model_log_likelihood(
            tree=tree, mutation_rate=self._mutation_rate
        )

    @property
    def penalized_log_likelihood(self):
        """
        The penalized log-likelihood of the training data.
        """
        return self._penalized_log_likelihood

    @property
    def log_likelihood(self):
        """
        The log-likelihood of the training data.
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

    @property
    def pseudo_mutations_per_edge(self):
        """
        The pseudo_mutations_per_edge.
        """
        return self._pseudo_mutations_per_edge

    @property
    def pseudo_non_mutations_per_edge(self):
        """
        The pseudo_non_mutations_per_edge.
        """
        return self._pseudo_non_mutations_per_edge

    @staticmethod
    def _get_long_edge_mutations(tree) -> Dict[Tuple[str, str], int]:
        """
        Mutations mapped across multiple edges.
        """
        long_edge_mutations = defaultdict(float)
        # We pre-compute all states since we will need repeated access
        character_states_dict = {
            node: tree.get_character_states(node) for node in tree.nodes
        }
        k = tree.character_matrix.shape[1]
        for node in tree.nodes:
            if tree.is_root(node):
                continue
            parent = tree.parent(node)
            character_states = character_states_dict[node]
            parent_states = character_states_dict[parent]
            for i in range(k):
                if (
                    character_states[i] > 0
                    and parent_states[i] == tree.missing_state_indicator
                ):
                    # Need to go up the tree and determine if we have a long
                    # edge mutation.
                    u = parent
                    while (
                        character_states_dict[u][i]
                        == tree.missing_state_indicator
                    ):
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
        return long_edge_mutations

    @staticmethod
    def model_log_likelihood(
        tree: CassiopeiaTree, mutation_rate: float
    ) -> float:
        """
        Model log-likelihood.

        The log-likelihood of the given character states under the model,
        up to constants (the q distribution is ignored).

        Used for cross-validation.
        """
        # # # # # Compute the log-likelihood for edges # # # # #
        log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = tree.get_time(child) - tree.get_time(parent)
            if edge_length < 0:
                raise ValueError("tree has negative branch lengths!")
            num_unmutated = len(
                tree.get_unmutated_characters_along_edge(parent, child)
            )
            num_mutated = len(
                tree.get_mutations_along_edge(
                    parent, child, treat_missing_as_mutations=False
                )
            )
            log_likelihood += num_unmutated * (-edge_length * mutation_rate)
            if num_mutated > 0:
                if edge_length * mutation_rate < 1e-8:
                    return -np.inf
                log_likelihood += num_mutated * np.log(
                    1 - np.exp(-edge_length * mutation_rate)
                )

        # # # # # Add in log-likelihood of long-edge mutations # # # # #
        long_edge_mutations = IIDExponentialMLE._get_long_edge_mutations(tree)
        for ((parent, child), num_mutated) in long_edge_mutations.items():
            edge_length = tree.get_time(child) - tree.get_time(parent)
            log_likelihood += num_mutated * np.log(
                1 - np.exp(-edge_length * mutation_rate)
            )

        if np.isnan(log_likelihood):
            raise ValueError("tree has nan log-likelihood.")
        return log_likelihood
