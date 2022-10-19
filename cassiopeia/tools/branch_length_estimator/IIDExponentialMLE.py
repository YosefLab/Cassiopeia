"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
from collections import defaultdict
from typing import List, Optional

import cvxpy as cp
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import IIDExponentialMLEError

from .BranchLengthEstimator import BranchLengthEstimator


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

    The estimated mutation rate(s) will be stored as an attribute called
    `mutation_rate`. The log-likelihood will be stored in an attribute
    called `log_likelihood`.

    Missing states are treated as missing at random by the model.

    Args:
        minimum_branch_length: Estimated branch lengths will be constrained to
            have length at least this value. By default it is set to 0.01,
            since the MLE tends to collapse mutationless edges to length 0.
        relative_mutation_rates: List of positive floats of length equal to the
            number of character sites. Number at each character site indicates
            the relative mutation rate at that site. Must be fully specified or
            None in which case all sites are assumed to evolve at the same rate.
            None is the default value for this argument.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
        verbose: Verbosity level.

    Attributes:
        mutation_rate: The estimated CRISPR/Cas9 mutation rate, assuming that
            the tree has depth exactly 1.
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        relative_mutation_rates: Optional[List[float]] = None,
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
        self._relative_mutation_rates = relative_mutation_rates
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
        relative_mutation_rates = self._relative_mutation_rates
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

        # # # # # Check that the relative_mutation_rates list is valid # # # # #
        is_rates_specified = False
        if relative_mutation_rates is not None:
            is_rates_specified = True
            if tree.character_matrix.shape[1] != len(relative_mutation_rates):
                raise ValueError(
                    "The number of character sites does not match the length \
                    of the provided relative_mutation_rates list. Please check \
                    your data."
                )
            for x in relative_mutation_rates:
                if x <= 0:
                    raise ValueError(
                        f"Relative mutation rates must be strictly positive, \
                        but you provided: {relative_mutation_rates}"
                    )
        else:
            relative_mutation_rates = [1.0] * tree.character_matrix.shape[1]

        # Group together sites having the same rate
        sites_by_rate = defaultdict(list)
        for i in range(len(relative_mutation_rates)):
            rate = relative_mutation_rates[i]
            sites_by_rate[rate].append(i)

        # # # # # Create variables of the optimization problem # # # # #
        t_variables = dict(
            [
                (node_id, cp.Variable(name=f"t_{node_id}"))
                for node_id in tree.nodes
            ]
        )

        # # # # # Create constraints of the optimization problem # # # # #
        a_leaf = tree.leaves[0]
        root = tree.root
        root_has_time_0_constraint = [t_variables[root] == 0]
        minimum_branch_length_constraints = [
            t_variables[child]
            >= t_variables[parent] + minimum_branch_length * t_variables[a_leaf]
            for (parent, child) in tree.edges
        ]
        ultrametric_constraints = [
            t_variables[leaf] == t_variables[a_leaf]
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
            edge_length = t_variables[child] - t_variables[parent]
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            for rate in sites_by_rate.keys():
                num_mutated = 0
                num_unmutated = 0
                for site in sites_by_rate[rate]:
                    if parent_states[site] == 0 and child_states[site] == 0:
                        num_unmutated += 1
                    elif parent_states[site] != child_states[site]:
                        if (
                            parent_states[site] != tree.missing_state_indicator
                            and child_states[site]
                            != tree.missing_state_indicator
                        ):
                            num_mutated += 1
                if num_unmutated > 0:
                    log_likelihood += num_unmutated * (-edge_length * rate)
                if num_mutated > 0:
                    log_likelihood += num_mutated * cp.log(
                        1 - cp.exp(-edge_length * rate - 1e-5)
                    )

        # # # # # Solve the problem # # # # #
        obj = cp.Maximize(log_likelihood)
        prob = cp.Problem(obj, all_constraints)
        try:
            prob.solve(solver=solver, verbose=verbose)
        except cp.SolverError:  # pragma: no cover
            raise IIDExponentialMLEError("Third-party solver failed")

        # # # # # Extract the mutation rate # # # # #
        scaling_factor = float(t_variables[a_leaf].value)
        if scaling_factor < 1e-8 or scaling_factor > 15.0:
            raise IIDExponentialMLEError(
                "The solver failed when it shouldn't have."
            )
        if is_rates_specified:
            self._mutation_rate = tuple(
                [rate * scaling_factor for rate in relative_mutation_rates]
            )
        else:
            self._mutation_rate = scaling_factor

        # # # # # Extract the log-likelihood # # # # #
        log_likelihood = float(log_likelihood.value)
        if np.isnan(log_likelihood):
            log_likelihood = -np.inf
        self._log_likelihood = log_likelihood

        # # # # # Populate the tree with the estimated branch lengths # # # # #
        times = {
            node: float(t_variables[node].value) / scaling_factor
            for node in tree.nodes
        }
        # Make sure that the root has time 0 (avoid epsilons)
        times[tree.root] = 0.0
        # We smooth out epsilons that might make a parent's time greater
        # than its child (which can happen if minimum_branch_length=0)
        for (parent, child) in tree.depth_first_traverse_edges():
            times[child] = max(times[parent], times[child])
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
        The estimated CRISPR/Cas9 mutation rate(s) under the given model. If
        relative_mutation_rates is specified, we return a list of rates (one per
        site). Otherwise all sites have the same rate and that rate is returned.
        """
        return self._mutation_rate
