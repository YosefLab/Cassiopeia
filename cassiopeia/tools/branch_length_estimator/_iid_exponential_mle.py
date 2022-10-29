"""
This file stores a subclass of BranchLengthEstimator, the IIDExponentialMLE.
Briefly, this model assumes that CRISPR/Cas9 mutates each site independently
and identically, with an exponential waiting time.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

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
    non-mutations can be added to regularize the MLE.

    The relative depth of each leaf can be specified to relax the ultrametric
    constraint. Also, the identicality assumption can be relaxed by providing
    the relative mutation rate of each site.

    The MLE under this set of assumptions is a special kind of convex
    optimization problem known as an exponential cone program, which can be
    readily solved with off-the-shelf (open source) solvers.

    Ancestral states may or may not all be provided. We recommend imputing them
    using the cassiopeia.tools.conservative_maximum_parsimony function.

    Missing states are treated as missing always completely at random (MACAR) by
    the model.

    The estimated mutation rate(s) will be stored as an attribute called
    `mutation_rate`. The log-likelihood will be stored as an attribute
    called `log_likelihood`. The penalized log-likelihood will be stored as an
    attribute called `penalized_log_likelihood` (the penalized log-likelihood
    includes the pseudocounts, whereas the log-likelihood does not).

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
        relative_mutation_rates: List of positive floats of length equal to the
            number of character sites. Number at each character site indicates
            the relative mutation rate at that site. Must be fully specified or
            None in which case all sites are assumed to evolve at the same rate.
            None is the default value for this argument.
        solver: Convex optimization solver to use. Can be "SCS", "ECOS", or
            "MOSEK". Note that "MOSEK" solver should be installed separately.
        verbose: Verbosity level.

    Attributes:
        mutation_rate: The estimated CRISPR/Cas9 mutation rate(s), assuming that
            the tree has depth exactly 1.
        log_likelihood: The log-likelihood of the training data under the
            estimated model.
        penalized_log_likelihood: The penalized log-likelihood (i.e., with
            pseudocounts) of the training data under the estimated model.
        minimum_branch_length: The minimum branch length (which was provided
            during initialization).
        pseudo_mutations_per_edge: The number of fictitious mutations added to
            each edge to regularize the MLE (which was provided during
            initialization).
        pseudo_non_mutations_per_edge: The number of fictitious non-mutations
            added to each edge to regularize the MLE (which was provided during
            initialization).
    """

    def __init__(
        self,
        minimum_branch_length: float = 0.01,
        pseudo_mutations_per_edge: float = 0.0,
        pseudo_non_mutations_per_edge: float = 0.0,
        relative_leaf_depth: Optional[List[Tuple[str, float]]] = None,
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
        self._pseudo_mutations_per_edge = pseudo_mutations_per_edge
        self._pseudo_non_mutations_per_edge = pseudo_non_mutations_per_edge
        self._relative_leaf_depth = relative_leaf_depth
        self._relative_mutation_rates = relative_mutation_rates
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

        # # # # # Check that the minimum_branch_length makes sense # # # # #
        if tree.get_edge_depth() * minimum_branch_length >= 1.0:
            raise ValueError(
                "The minimum_branch_length is too large. Please reduce it."
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

        # # # # # Get and check relative_leaf_depth # # # # #
        if relative_leaf_depth is None:
            relative_leaf_depth = [(leaf, 1.0) for leaf in tree.leaves]
        if sorted([leaf for (leaf, _) in relative_leaf_depth]) != sorted(
            tree.leaves
        ):
            raise ValueError(
                "All leaves - and only leaves - must be specified in "
                f"relative_leaf_depth. You provided: relative_leaf_depth = "
                f"{relative_leaf_depth} but the leaves in the tree are: "
                f"{tree.leaves}"
            )
        deepest_leaf = sorted(
            [
                (relative_depth, leaf)
                for (leaf, relative_depth) in relative_leaf_depth
            ]
        )[-1][1]
        relative_leaf_depth = dict(relative_leaf_depth)

        # # # # # Create variables of the optimization problem # # # # #
        t_variables = dict(
            [
                (node_id, cp.Variable(name=f"t_{node_id}"))
                for node_id in tree.nodes
            ]
        )

        # # # # # Create constraints of the optimization problem # # # # #
        root = tree.root
        root_has_time_0_constraint = [t_variables[root] == 0]
        minimum_branch_length_constraints = [
            t_variables[child]
            >= t_variables[parent]
            + minimum_branch_length * t_variables[deepest_leaf]
            for (parent, child) in tree.edges
        ]
        ultrametric_constraints = [
            t_variables[leaf]
            == t_variables[deepest_leaf]
            * relative_leaf_depth[leaf]
            / relative_leaf_depth[deepest_leaf]
            for leaf in tree.leaves
            if leaf != deepest_leaf
        ]
        all_constraints = (
            root_has_time_0_constraint
            + minimum_branch_length_constraints
            + ultrametric_constraints
        )

        # # # # # Compute the penalized log-likelihood for edges # # # # #
        penalized_log_likelihood = 0
        num_sites = tree.character_matrix.shape[1]
        assert (
            sum([len(sites_by_rate[rate]) for rate in sites_by_rate.keys()])
            == num_sites
        )
        for (parent, child) in tree.edges:
            edge_length = t_variables[child] - t_variables[parent]
            parent_states = tree.get_character_states(parent)
            child_states = tree.get_character_states(child)
            for rate in sites_by_rate.keys():
                num_mutated = (
                    pseudo_mutations_per_edge
                    * len(sites_by_rate[rate])
                    / num_sites
                )
                num_unmutated = (
                    pseudo_non_mutations_per_edge
                    * len(sites_by_rate[rate])
                    / num_sites
                )
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
                    penalized_log_likelihood += num_unmutated * (
                        -edge_length * rate
                    )
                if num_mutated > 0:
                    penalized_log_likelihood += num_mutated * cp.log(
                        1 - cp.exp(-edge_length * rate - 1e-5)
                    )

        # # # # # Add in log-likelihood of long-edge mutations # # # #
        long_edge_mutations = self._get_long_edge_mutations(tree, sites_by_rate)
        for rate in long_edge_mutations:
            for ((parent, child), num_mutated) in long_edge_mutations[
                rate
            ].items():
                edge_length = t_variables[child] - t_variables[parent]
                penalized_log_likelihood += num_mutated * cp.log(
                    1
                    - cp.exp(
                        -edge_length * rate - 1e-5
                    )  # We add eps for stability.
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
        scaling_factor = float(t_variables[deepest_leaf].value)
        if scaling_factor < 1e-8 or scaling_factor > 15.0:
            # Note: when passing in very small relative mutation rates, this
            # check will fail even though everything is OK. Still worth checking
            # and raising an error.
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
        The estimated CRISPR/Cas9 mutation rate(s) under the given model. If
        relative_mutation_rates is specified, we return a list of rates (one per
        site). Otherwise all sites have the same rate and that rate is returned.
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
    def _get_long_edge_mutations(
        tree,
        sites_by_rate: Dict[float, List[int]],
    ) -> Dict[float, Dict[Tuple[str, str], int]]:
        """
        Mutations mapped across multiple edges, by rate.
        """
        long_edge_mutations = {
            rate: defaultdict(float) for rate in sites_by_rate.keys()
        }
        # We pre-compute all states since we will need repeated access
        character_states_dict = {
            node: tree.get_character_states(node) for node in tree.nodes
        }
        for node in tree.nodes:
            if tree.is_root(node):
                continue
            parent = tree.parent(node)
            character_states = character_states_dict[node]
            parent_states = character_states_dict[parent]
            for rate in sites_by_rate.keys():
                for i in sites_by_rate[rate]:
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
                            long_edge_mutations[rate][(u, node)] += 1
                        else:
                            if (
                                character_states_dict[u][i]
                                != character_states[i]
                            ):
                                raise Exception(
                                    "Ancestral state reconstruction seems "
                                    f"invalid: character {character_states[i]} "
                                    "descends from "
                                    f"{character_states_dict[u][i]}."
                                )
                    elif character_states[i] == 0 and parent_states[i] != 0:
                        raise Exception(
                            "If a node has state 0 (uncut), its parent should "
                            "also have state 0."
                        )
        return long_edge_mutations

    @staticmethod
    def model_log_likelihood(
        tree: CassiopeiaTree, mutation_rate: Union[float, List[float]]
    ) -> float:
        """
        Model log-likelihood.

        The log-likelihood of the given character states under the model,
        up to constants (the q distribution is ignored).

        Used for cross-validation.

        Args:
            tree: The given tree with branch lengths
            mutation_rate: Either the mutation rate of all sites (a float) or a
                list of mutation rates, one per site.
        """
        num_sites = tree.character_matrix.shape[1]
        if type(mutation_rate) is float:
            mutation_rate = [mutation_rate] * num_sites
        if len(mutation_rate) != tree.character_matrix.shape[1]:
            raise ValueError(
                "mutation_rate must have the same length as the number of "
                f"sites in the tree, but mutation_rate = {mutation_rate} "
                f"whereas the tree has {num_sites} sites."
            )

        # Group together sites having the same rate
        sites_by_rate = defaultdict(list)
        for i in range(len(mutation_rate)):
            rate = mutation_rate[i]
            sites_by_rate[rate].append(i)

        # # # # # Compute the log-likelihood for edges # # # # #
        log_likelihood = 0
        for (parent, child) in tree.edges:
            edge_length = tree.get_time(child) - tree.get_time(parent)
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
                    log_likelihood += num_mutated * np.log(
                        1 - np.exp(-edge_length * rate - 1e-5)
                    )

        # # # # # Add in log-likelihood of long-edge mutations # # # # #
        long_edge_mutations = IIDExponentialMLE._get_long_edge_mutations(
            tree, sites_by_rate
        )
        for rate in long_edge_mutations:
            for ((parent, child), num_mutated) in long_edge_mutations[
                rate
            ].items():
                edge_length = tree.get_time(child) - tree.get_time(parent)
                log_likelihood += num_mutated * np.log(
                    1
                    - np.exp(
                        -edge_length * rate - 1e-5
                    )  # We add eps for stability.
                )

        if np.isnan(log_likelihood):
            raise ValueError("tree has nan log-likelihood.")

        return log_likelihood
