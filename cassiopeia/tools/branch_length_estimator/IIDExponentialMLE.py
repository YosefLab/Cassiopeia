"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
import cvxpy as cp
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import IIDExponentialMLEError

from .BranchLengthEstimator import BranchLengthEstimator


class IIDExponentialMLE(BranchLengthEstimator):
    """
    MLE under a model of IID memoryless CRISPR/Cas9 mutations.

    This model assumes that CRISPR/Cas9 mutates each site independently and
    identically, with an exponential waiting time. The MLE under this model
    is a special kind of convex optimization problem known as an exponential
    cone program, which can be readily solved with off-the-shelf (open source)
    solvers.

    This estimator requires that the ancestral characters be provided (these
    can be imputed with CassiopeiaTree's reconstruct_ancestral_characters
    method if they are not known, which is usually the case for real data).

    Because branch lengths and CRISPR/Cas9 mutation rate are not identifiable,
    this estimator assumes that the tree has a depth of 1. In other words,
    the estimated tree will have depth 1. The estimated mutation rate under
    this unit-depth assumption will be stored as an attribute called
    `mutation_rate`.

    Missing states are treated as missing at random by the model.

    Args:
        minimum_branch_length: Estimated branch lengths will be constrained to
            have length at least this value. By default it is set to 0.01,
            since the MLE tends to collapse mutationless edges to length 0.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
        verbose: Verbosity level.

    Attributes:
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        mutation_rate: The estimated CRISPR/Cas9 mutation rate, assuming that
            the tree has depth exactly 1.
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        verbose: bool = False,
        solver: str = "SCS",
    ):
        allowed_solvers = ["ECOS", "SCS", "MOSEK"]
        if solver not in allowed_solvers:
            raise ValueError(
                f"Solver {solver} not allowed. Allowed solvers: {allowed_solvers}"
            )  # pragma: no cover
        self._minimum_branch_length = minimum_branch_length
        self._verbose = verbose
        self._solver = solver

    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        See base class. The only caveat is that this method raises an
        IIDExponentialMLEError if the underlying convex optimization
        solver fails.

        Raises:
            IIDExponentialMLEError
        """
        # Extract parameters
        minimum_branch_length = self._minimum_branch_length
        solver = self._solver
        verbose = self._verbose

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

        # # # # # Compute the log-likelihood # # # # #
        log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = r_X_t_variables[child] - r_X_t_variables[parent]
            num_unmutated = len(
                tree.get_unmutated_characters_along_edge(parent, child)
            )
            num_mutated = len(
                tree.get_mutated_characters_along_edge(parent, child)
            )
            log_likelihood += num_unmutated * (-edge_length)
            log_likelihood += num_mutated * cp.log(
                1 - cp.exp(-edge_length - 1e-5)  # We add eps for stability.
            )

        # # # # # Solve the problem # # # # #
        obj = cp.Maximize(log_likelihood)
        prob = cp.Problem(obj, all_constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except:  # pragma: no cover
            raise IIDExponentialMLEError("Third-party solver failed")

        # # # # # Populate the tree with the estimated branch lengths # # # # #
        times = {
            node: float(r_X_t_variables[node].value) for node in tree.nodes
        }
        # Make sure that the root has time 0 (avoid epsilons)
        times[tree.root] = 0.0
        # We smooth out epsilons that might make a parent's time greater
        # than its child (which can happen if minimum_branch_length=0)
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
        tree.set_times(times)

        # # # # # Extract log-likelihood # # # # #
        log_likelihood = float(log_likelihood.value)
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
        self._log_likelihood = log_likelihood

        # # # # # Extract mutation rate # # # # #
        tree_depth = tree.get_depth()
        self._mutation_rate = tree_depth

        # # # # # Raise errors on border cases # # # # #
        if tree_depth < 1e-6:
            raise IIDExponentialMLEError(
                "All branch lengths estimated as zero."
            )
        if tree_depth > 15.0:
            raise IIDExponentialMLEError(
                "Branch lengths estimated as infinite."
            )
        
        # # # # # Make tree have depth 1 # # # # #
        tree.scale_to_unit_length()

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def mutation_rate(self):
        return self._mutation_rate
