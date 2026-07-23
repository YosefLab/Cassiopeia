"""File containing functions for scoring metrics on a tree."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import combinations

import networkx as nx
import numpy as np
import scipy
import scipy.stats
from treedata import TreeData

from cassiopeia.mixins import TreeMetricError
from cassiopeia.tools import parameter_estimators
from cassiopeia.tools.ancestral_characters import _seed_leaf_states, ancestral_characters
from cassiopeia.tools.topology import count_edge_mutations, get_leaves, get_root
from cassiopeia.utils import (
    _get_characters,
    _get_digraph,
    _get_parameter,
    _get_root,
)


def _resolve_missing_state(tdata: TreeData, missing_state: int | str | None = None):
    """Resolve a single missing-state sentinel value from a TreeData object."""
    value = _get_parameter(tdata, "missing_state", value=missing_state)
    if isinstance(value, (list, tuple, set)):
        return next(iter(value))
    return value


def _branch_length(g: nx.DiGraph, parent: str, child: str) -> float:
    """Return the branch length of an edge, defaulting to 1.0 when unset."""
    return g.edges[parent, child].get("length", 1.0)


def calculate_parsimony(
    tdata: TreeData,
    tree_key: str | None = None,
    characters_key: str = "characters",
    infer_ancestral_characters: bool = False,
    treat_missing_as_mutation: bool = False,
    missing_state: str | int | None = None,
    unmodified_state: str | int | None = None,
    key_added: str = "n_mutations",
) -> int:
    """Calculate the number of mutations that have occurred on a tree.

    Calculates the parsimony, defined as the number of character/state
    mutations that occur on edges of the tree, from the character state
    annotations at the nodes. A mutation is said to have occurred on an
    edge if a state is present at a character at the child node and this
    state is not in the parent node.

    If ``infer_ancestral_characters`` is set to ``True``, the internal nodes'
    character states are first inferred by Camin-Sokal parsimony from the leaf
    states (see :func:`cassiopeia.tl.ancestral_characters`). Otherwise, the
    existing node annotations under ``characters_key`` are used. The per-edge
    mutation counts are written under the ``key_added`` edge attribute (see
    :func:`cassiopeia.tl.count_edge_mutations`) and summed.

    Args:
        tdata: TreeData object to operate on.
        tree_key: The ``obst`` key of the tree to use.
        characters_key: Node attribute holding character states.
        infer_ancestral_characters: Whether to infer the ancestral character
            states of the tree before counting mutations.
        treat_missing_as_mutation: Whether to count transitions to a missing
            state as mutations.
        missing_state: Missing-data value. Resolved from ``tdata`` when ``None``.
        unmodified_state: Unmodified state value. Resolved from ``tdata`` when
            ``None``. Only used when inferring ancestral characters.
        key_added: Edge attribute under which per-edge mutation counts are
            stored.

    Returns:
        The number of mutations that have occurred on the tree.

    Raises:
        CassiopeiaError: If a node does not have character states and they are
            not being inferred.
    """
    if infer_ancestral_characters:
        ancestral_characters(
            tdata,
            characters_key=characters_key,
            tree_key=tree_key,
            missing_state=missing_state,
            unmodified_state=unmodified_state,
        )

    count_edge_mutations(
        tdata,
        tree_key=tree_key,
        characters_key=characters_key,
        treat_missing_as_mutation=treat_missing_as_mutation,
        missing_state=missing_state,
        key_added=key_added,
    )

    g, _ = _get_digraph(tdata, tree_key)
    return int(sum(g.edges[u, v][key_added] for u, v in g.edges))


def log_transition_probability(
    character: int,
    priors: dict[int, dict[int, float]],
    missing_state: int | str,
    s: int | str,
    s_: int | str,
    t: float,
    mutation_probability_function_of_time: Callable[[float], float],
    missing_probability_function_of_time: Callable[[float], float],
) -> float:
    """Gives the log transition probability between two given states.

    Assumes that 0 is the uncut-state, and that only 0 has non-zero probability
    of transitioning to non-0 states. Additionally, assumes that any non-missing
    state can mutate to the missing state, specified by ``missing_state``.

    Here, "&" represents a placeholder for any non-missing state. Thus, the
    probability of transitioning from any state to any given non-missing
    state is (1 - probability of transitioning to the missing state).

    The probability of acquiring a mutation is given by a time t and
    ``mutation_probability_function_of_time``, and the same is true of heritable
    missing data events and ``missing_probability_function_of_time``. In
    determining the probability of acquiring a non-missing, non-0 state, the
    ``priors`` are used.

    Args:
        character: The character whose distribution to draw the prior from.
        priors: Priors for character states, mapping character index to a dict
            mapping state to prior probability.
        missing_state: The missing-data sentinel value.
        s: The original state.
        s_: The state being transitioned to.
        t: The length of time that the transition can occur along.
        mutation_probability_function_of_time: The function defining the
            probability of a lineage acquiring a mutation within a given time.
        missing_probability_function_of_time: The function defining the
            probability of a lineage acquiring heritable missing data within a
            given time.

    Returns:
        The log transition probability between the states.
    """
    # A probability of exactly 0 (e.g. a certain mutation, so P(no mutation) = 0)
    # legitimately yields a log-probability of -inf; suppress numpy's spurious
    # "divide by zero encountered in log" warning for that intended result.
    with np.errstate(divide="ignore"):
        if s_ == missing_state:
            if s == missing_state:
                return 0
            else:
                return np.log(missing_probability_function_of_time(t))
        # "&" stands in for any non-missing state (including the uncut state).
        # The sum probability of transitioning from any non-missing state s
        # to any non-missing state s' is 1 - P(missing event). Used to avoid
        # marginalizing over the entire state space.
        elif s_ == "&":
            if s == missing_state:
                return -1e16
            else:
                return np.log(1 - missing_probability_function_of_time(t))
        elif s_ == 0:
            if s == 0:
                return np.log(1 - mutation_probability_function_of_time(t)) + np.log(
                    1 - missing_probability_function_of_time(t)
                )
            else:
                # The transition from "&" to a non-missing state cannot occur
                return -1e16
        else:
            if s == missing_state:
                return -1e16
            elif s == 0:
                return (
                    np.log(mutation_probability_function_of_time(t))
                    + np.log(priors[character][s_])
                    + np.log(1 - missing_probability_function_of_time(t))
                )
            elif s == s_:
                return np.log(1 - missing_probability_function_of_time(t))
            else:
                # The transition from "&" to a non-missing state cannot occur
                return -1e16


def log_likelihood_of_character(
    g: nx.DiGraph,
    root: str,
    character: int,
    priors: dict[int, dict[int, float]],
    missing_state: int | str,
    characters_key: str,
    use_internal_character_states: bool,
    mutation_probability_function_of_time: Callable[[float], float],
    missing_probability_function_of_time: Callable[[float], float],
    stochastic_missing_probability: float,
    implicit_root_branch_length: float,
) -> float:
    """Calculates the log likelihood of a given character on the tree.

    Calculates the log likelihood of a tree given the states at a given
    character in the leaves using Felsenstein's Pruning Algorithm, which sets
    up a recursive relation between the likelihoods of states at nodes for this
    character. The likelihood L(s, n) at a given state s at a given node n is:

    L(s, n) = Π_{n'}(Σ_{s'}(P(s'|s) * L(s', n')))

    for all n' that are children of n, and s' in the state space, with
    P(s'|s) being the transition probability from s to s'. This includes the
    missing state, specified by ``missing_state``.

    We assume here that mutations are irreversible. The user can choose to use
    the character states annotated at internal nodes; if not, the likelihood is
    marginalized over all possible internal state characters, assuming the root
    has the unmutated state at each character with an implicit branch of length
    ``implicit_root_branch_length`` leading into it.

    Args:
        g: The tree graph, with character states stored under the
            ``characters_key`` node attribute.
        root: The root node of *g*.
        character: The index of the character to calculate the likelihood of.
        priors: Priors for character states.
        missing_state: The missing-data sentinel value.
        characters_key: Node attribute holding character states.
        use_internal_character_states: Indicates if internal node character
            states should be assumed to be specified exactly.
        mutation_probability_function_of_time: The function defining the
            probability of a lineage acquiring a mutation within a given time.
        missing_probability_function_of_time: The function defining the
            probability of a lineage acquiring heritable missing data within a
            given time.
        stochastic_missing_probability: The probability that a cell/character
            pair acquires stochastic missing data at the end of the lineage.
        implicit_root_branch_length: The length of the implicit root branch,
            used if the implicit root needs to be added.

    Returns:
        The log likelihood of the tree on one character.
    """

    def is_leaf(n):
        return g.out_degree(n) == 0

    # This dictionary uses a nested dictionary structure. Each node is mapped
    # to a dictionary storing the likelihood for each possible state
    # (states that have non-0 likelihood)
    likelihoods_at_nodes = {}

    # Perform a DFS to propagate the likelihood from the leaves
    for n in nx.dfs_postorder_nodes(g, source=root):
        # If states are observed, their likelihoods are set to 1
        if is_leaf(n):
            likelihoods_at_nodes[n] = {g.nodes[n][characters_key][character]: 0}
            continue

        possible_states = []
        # If internal character states are to be used, then the likelihood
        # for all other states are ignored. Otherwise, marginalize over
        # only states that do not break irreversibility, as all states that
        # do have likelihood of 0
        if use_internal_character_states:
            possible_states = [g.nodes[n][characters_key][character]]
        else:
            child_possible_states = []
            for c in [set(likelihoods_at_nodes[child]) for child in g.successors(n)]:
                if missing_state not in c and "&" not in c:
                    child_possible_states.append(c)
            # "&" stands in for any non-missing state (including uncut), and
            # is a possible state when all children are missing, as any
            # state could have occurred at the parent if all missing data
            # events occurred independently. Used to avoid marginalizing
            # over the entire state space.
            if child_possible_states == []:
                possible_states = [
                    "&",
                    missing_state,
                ]
            else:
                possible_states = list(set.intersection(*child_possible_states))
                if 0 not in possible_states:
                    possible_states.append(0)

        # This stores the likelihood of each possible state at the current node
        likelihoods_per_state_at_n = {}

        # We calculate the likelihood of the states at the current node
        # according to the recurrence relation. For each state, we marginalize
        # over the likelihoods of the states that it could transition to in the
        # daughter nodes
        for s in possible_states:
            likelihood_for_s = 0
            for child in g.successors(n):
                likelihoods_for_s_marginalize_over_s_ = []
                for s_ in likelihoods_at_nodes[child]:
                    likelihood_s_ = (
                        log_transition_probability(
                            character,
                            priors,
                            missing_state,
                            s,
                            s_,
                            _branch_length(g, n, child),
                            mutation_probability_function_of_time,
                            missing_probability_function_of_time,
                        )
                        + likelihoods_at_nodes[child][s_]
                    )
                    # Here we take into account the probability of
                    # stochastic missing data
                    if is_leaf(child):
                        if s_ == missing_state and s != missing_state:
                            likelihood_s_ = np.log(
                                np.exp(likelihood_s_)
                                + (
                                    1
                                    - missing_probability_function_of_time(
                                        _branch_length(g, n, child)
                                    )
                                )
                                * stochastic_missing_probability
                            )
                        if s_ != missing_state:
                            likelihood_s_ += np.log(1 - stochastic_missing_probability)
                    likelihoods_for_s_marginalize_over_s_.append(likelihood_s_)
                likelihood_for_s += scipy.special.logsumexp(
                    np.array(likelihoods_for_s_marginalize_over_s_)
                )
            likelihoods_per_state_at_n[s] = likelihood_for_s

        likelihoods_at_nodes[n] = likelihoods_per_state_at_n

    # If we are not to use the internal state annotations explicitly,
    # then we assume an implicit root where each state is the uncut state (0)
    # Thus, we marginalize over the transition from 0 in the implicit root
    # to all non-0 states in its child
    if not use_internal_character_states:
        # If the implicit root does not exist in the tree, then we impose it,
        # with the length of the branch being specified as
        # `implicit_root_branch_length`. Otherwise, we just use the existing
        # root with a singleton child as the implicit root
        if g.out_degree(root) != 1:
            likelihood_contribution_from_each_root_state = [
                log_transition_probability(
                    character,
                    priors,
                    missing_state,
                    0,
                    s_,
                    implicit_root_branch_length,
                    mutation_probability_function_of_time,
                    missing_probability_function_of_time,
                )
                + likelihoods_at_nodes[root][s_]
                for s_ in likelihoods_at_nodes[root]
            ]
            likelihood_at_implicit_root = scipy.special.logsumexp(
                likelihood_contribution_from_each_root_state
            )

            return likelihood_at_implicit_root

        else:
            # Here we account for the edge case in which all of the leaves are
            # missing, in which case the root will have "&" in place of 0. The
            # likelihood at "&" will have the same likelihood as 0 based on the
            # transition rules regarding "&". As "&" is a placeholder when the
            # state is unknown, this can be thought of realizing "&" as 0.
            if 0 not in likelihoods_at_nodes[root]:
                return likelihoods_at_nodes[root]["&"]
            else:
                # Otherwise, we return the likelihood of the 0 state at the
                # existing implicit root
                return likelihoods_at_nodes[root][0]

    # If we use the internal state annotations explicitly, then we return
    # the likelihood of the state annotated at this character at the root
    else:
        return list(likelihoods_at_nodes[root].values())[0]


def get_tracing_parameters(
    tdata: TreeData,
    continuous: bool,
    assume_root_implicit_branch: bool,
    characters_key: str = "characters",
    time_key: str = "time",
    tree_key: str | None = None,
) -> tuple[float, float, float]:
    """Gets the lineage tracing parameters from a tree.

    This function attempts to consume these parameters from ``tdata.uns`` as
    ``mutation_rate``, ``heritable_missing_rate``, and
    ``stochastic_missing_probability``. If the rates are not found, they are
    estimated using their respective estimators. Note that in order to estimate
    the missing data parameters, at least one of the two must be populated.

    Args:
        tdata: TreeData object on which to consume/estimate the parameters.
        continuous: If the parameters are to be estimated, whether to estimate
            them as continuous or discrete parameters.
        assume_root_implicit_branch: Whether to include an implicit root branch
            in the tree depth/time when estimating the rate parameters.
        characters_key: The ``obsm`` key for the character matrix used to
            estimate parameters.
        time_key: Node attribute key containing depth/time values.
        tree_key: The ``obst`` key of the tree to use.

    Returns:
        The mutation rate, the heritable missing rate, and the stochastic
        missing probability.

    Raises:
        TreeMetricError: If one of the provided/estimated parameters is invalid.
    """
    mutation_rate = _get_parameter(tdata, "mutation_rate")
    if mutation_rate is None:
        mutation_rate = parameter_estimators.estimate_mutation_rate(
            tdata,
            continuous,
            assume_root_implicit_branch,
            characters_key=characters_key,
            depth_key=time_key,
            tree_key=tree_key,
        )

    stochastic_missing_probability = _get_parameter(tdata, "stochastic_missing_probability")
    heritable_missing_rate = _get_parameter(tdata, "heritable_missing_rate")
    if not (stochastic_missing_probability is not None and heritable_missing_rate is not None):
        (
            stochastic_missing_probability,
            heritable_missing_rate,
        ) = parameter_estimators.estimate_missing_rates(
            tdata,
            continuous,
            assume_root_implicit_branch,
            characters_key=characters_key,
            depth_key=time_key,
            tree_key=tree_key,
        )

    # We check that the mutation and missing rates have valid values
    if mutation_rate < 0:
        raise TreeMetricError("Mutation rate must be > 0.")
    if not continuous and mutation_rate > 1:
        raise TreeMetricError("Per-generation mutation rate must be < 1.")
    if heritable_missing_rate < 0:
        raise TreeMetricError("Heritable missing data rate must be > 0.")
    if not continuous and heritable_missing_rate > 1:
        raise TreeMetricError("Per-generation heritable missing data rate must be < 1.")
    if stochastic_missing_probability < 0:
        raise TreeMetricError("Stochastic missing data rate must be > 0.")
    if stochastic_missing_probability > 1:
        raise TreeMetricError("Stochastic missing data rate must be < 1.")

    return mutation_rate, heritable_missing_rate, stochastic_missing_probability


def calculate_likelihood(
    tdata: TreeData,
    model: str = "discrete",
    use_internal_character_states: bool = False,
    characters_key: str = "characters",
    time_key: str = "time",
    tree_key: str | None = None,
) -> float:
    """Calculates the log likelihood of a tree under a lineage-tracing model.

    A wrapper for :func:`get_tracing_parameters` and
    :func:`log_likelihood_of_character`. The mutation rate, heritable missing
    rate, and stochastic missing probability are consumed from / estimated for
    the tree, and the per-character log likelihoods (assuming characters mutate
    independently) are summed.

    Two evolutionary models are supported:

    * ``'discrete'``: rates are per-generation, branch lengths are ignored, and
      the implicit root branch length is 1.
    * ``'continuous'``: rates are instantaneous; the waiting time until an event
      is exponentially distributed, so the probability that an event occurs in
      time ``t`` is given by the exponential CDF over branch lengths. The
      implicit root branch length is the mean branch length of the tree.

    Args:
        tdata: TreeData object on which to calculate the likelihood.
        model: The evolutionary model, either ``'discrete'`` or ``'continuous'``.
        use_internal_character_states: Indicates if internal node character
            states should be assumed to be specified exactly.
        characters_key: The ``obsm`` key for the character matrix (also the node
            attribute under which internal character states are read).
        time_key: Node attribute key containing depth/time values (used when
            estimating parameters).
        tree_key: The ``obst`` key of the tree to use.

    Returns:
        The log likelihood of the tree given the observed character states.

    Raises:
        TreeMetricError: If ``model`` is invalid, the priors are not populated,
            the consumed parameters are invalid, or character state annotations
            are missing at a node.
    """
    if model not in ("discrete", "continuous"):
        raise TreeMetricError(f"Unknown model {model!r}. Use 'discrete' or 'continuous'.")
    continuous = model == "continuous"

    priors = _get_parameter(tdata, "priors")
    if priors is None:
        raise TreeMetricError("Priors must be specified for this tree to calculate the likelihood.")

    missing_state = _resolve_missing_state(tdata)

    # TreeData stores frozen graphs; operate on a copy and seed leaf states.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)
    root = get_root(g)

    character_matrix = _get_characters(tdata, characters_key)
    _seed_leaf_states(g, character_matrix, characters_key)

    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    for leaf in leaves:
        if not g.nodes[leaf].get(characters_key):
            raise TreeMetricError(
                "Character states have not been initialized at leaves."
                " Seed leaf character states from a character matrix."
            )

    if use_internal_character_states:
        for n in g.nodes:
            if g.out_degree(n) == 0:
                continue
            if not g.nodes[n].get(characters_key):
                raise TreeMetricError(
                    "Character states empty at internal node. Character states"
                    " must be annotated at each node if internal character"
                    " states are to be used."
                )

    (
        mutation_rate,
        heritable_missing_rate,
        stochastic_missing_probability,
    ) = get_tracing_parameters(
        tdata,
        continuous,
        (not use_internal_character_states),
        characters_key=characters_key,
        time_key=time_key,
        tree_key=tree_key,
    )

    if continuous:
        mutation_probability_function_of_time = lambda t: 1 - np.exp(-mutation_rate * t)
        missing_probability_function_of_time = lambda t: 1 - np.exp(-heritable_missing_rate * t)
        implicit_root_branch_length = float(np.mean([_branch_length(g, u, v) for u, v in g.edges]))
    else:
        mutation_probability_function_of_time = lambda t: mutation_rate
        missing_probability_function_of_time = lambda t: heritable_missing_rate
        implicit_root_branch_length = 1

    n_character = len(g.nodes[leaves[0]][characters_key])

    return np.sum(
        [
            log_likelihood_of_character(
                g,
                root,
                character,
                priors,
                missing_state,
                characters_key,
                use_internal_character_states,
                mutation_probability_function_of_time,
                missing_probability_function_of_time,
                stochastic_missing_probability,
                implicit_root_branch_length,
            )
            for character in range(n_character)
        ]
    )


def _normalized_collision(state_priors: dict) -> float:
    """Return ``sum_s p_s^2`` for one character, renormalizing priors to sum to 1.

    KPTracer-style priors files store unnormalized allele-frequency weights
    (each character can sum to e.g. ~3.4 rather than 1). Squaring those values
    directly yields ``q > 1`` and corrupts the p-values, so the priors are
    renormalized first. For priors that already sum to 1 this is a no-op.
    """
    vals = np.array(list(state_priors.values()), dtype=float)
    total = vals.sum()
    if total <= 0:
        raise TreeMetricError("Prior weights for a character sum to <= 0; cannot normalize.")
    vals = vals / total
    return float(np.sum(vals**2))


def _collision_probability(
    priors: dict | None,
    character_matrix,
    missing_state: int | str,
    unmodified_state: int | str,
) -> float:
    """Estimate the state-collision probability ``q``.

    The collision probability is the chance that two independent mutations at a
    character produce the same state, i.e. ``sum_s p_s^2`` where ``p_s`` is the
    prior probability of state ``s``. When ``priors`` is a mapping of character
    index to per-state dicts, the per-character collisions are averaged. When it
    is a flat state->probability mapping, that single distribution is used. When
    no priors are available, a uniform distribution over the observed editable
    states is assumed, giving ``1 / m`` for ``m`` distinct non-missing,
    non-unmodified states.
    """
    if priors is None:
        warnings.warn(
            "Neither `collision_probability` nor `priors` were provided; "
            "assuming a uniform distribution (q = 1/m) over observed states.",
            UserWarning,
            stacklevel=3,
        )
        states = set(np.unique(character_matrix.values)) - {missing_state, unmodified_state}
        if not states:
            raise TreeMetricError("Character matrix contains no editable states.")
        return 1.0 / len(states)

    if not isinstance(priors, dict):
        raise TreeMetricError("`priors` must be a dict or a dict of per-character dicts.")

    if len(priors) == 0:
        raise TreeMetricError("`priors` is empty; cannot estimate collision probability.")

    first_value = next(iter(priors.values()))
    if isinstance(first_value, dict):
        per_character = [_normalized_collision(state_priors) for state_priors in priors.values()]
        return float(np.mean(per_character))
    return _normalized_collision(priors)


def _calculate_cphs(
    g: nx.DiGraph,
    mutation_rate: float,
    collision_probability: float,
    time_key: str,
    characters_key: str,
    missing_state: int | str,
    unmodified_state: int | str,
) -> float:
    """Calculate the corrected pairwise homoplasy score (cPHS) for one tree.

    Character states (including inferred ancestral states at internal nodes)
    must already be annotated under the ``characters_key`` node attribute.
    """
    root = _get_root(g)
    leaves = get_leaves(g)
    k = len(g.nodes[leaves[0]][characters_key])

    # cPHS models homoplasy as a function of the normalized height of a pair's
    # LCA, and so requires an ultrametric (equal leaf-depth) tree.
    leaf_depths = np.array([g.nodes[leaf][time_key] for leaf in leaves], dtype=float)
    if not np.allclose(leaf_depths, leaf_depths[0]):
        raise TreeMetricError(
            "All leaves must be at the same depth to calculate cPHS. Perform "
            "branch length estimation using `ConvexML` or `LAML-Pro`"
        )
    leaf_depth = leaf_depths[0]
    if leaf_depth == 0:
        raise TreeMetricError(
            f"Leaves have zero depth under node attribute {time_key!r}; cPHS "
            "requires branch length estimation."
        )

    # Homoplasy count and (normalized) LCA height for every pair of leaves. A
    # homoplasy at a character is a shared, non-missing edit in both leaves whose
    # LCA is still in the unmodified state (i.e. the edit arose independently).
    phs = []
    lca_heights = []
    for (l1, l2), lca in nx.tree_all_pairs_lowest_common_ancestor(
        g, root=root, pairs=combinations(leaves, 2)
    ):
        lca_states = g.nodes[lca][characters_key]
        l1_states = g.nodes[l1][characters_key]
        l2_states = g.nodes[l2][characters_key]
        phs.append(
            sum(
                1
                for i in range(k)
                if (
                    lca_states[i] == unmodified_state
                    and l1_states[i] not in (missing_state, unmodified_state)
                    and l1_states[i] == l2_states[i]
                )
            )
        )
        lca_heights.append(g.nodes[lca][time_key])
    phs = np.array(phs)
    lca_heights = np.array(lca_heights, dtype=float) / leaf_depth

    # Probability of a homoplasy at a given LCA height under the mutation model:
    # the LCA is unmodified (alpha), both descendant branches acquire an edit
    # (beta**2), and those edits collide on the same state (q).
    alpha = np.exp(-mutation_rate * lca_heights)
    beta = 1 - np.exp(-mutation_rate * (1 - lca_heights))
    prob = alpha * beta**2 * collision_probability
    prob[np.isclose(lca_heights, 1)] = 1
    pvalues = 1 - scipy.stats.binom.cdf(phs - 1, k, prob)
    pvalues[pvalues == 0] = np.finfo(float).eps  # zeros cannot be real zeros

    # Benjamini-Hochberg style adjustment; the cPHS is the minimum adjusted p-value.
    pvalues_sorted = np.sort(pvalues)
    adjusted_pvalues = pvalues_sorted * len(pvalues) / np.arange(1, len(pvalues) + 1)
    return float(np.min(adjusted_pvalues))


def calculate_cPHS(
    tdata: TreeData,
    characters_key: str = "characters",
    time_key: str = "time",
    mutation_rate: float | None = None,
    collision_probability: float | None = None,
    priors: dict | None = None,
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    tree_key: str | None = None,
) -> float | dict[str, float]:
    """Calculate the corrected Pairwise Homoplasy Score (cPHS) of a tree.

    Given a tree with inferred branch lengths and ancestral character states, the
    cPHS statistic uses a homoplasy-based approach to assess the accuracy of a
    tree reconstruction by quantifying the likelihood of the observed homoplasies
    under a mutation model (Zilber et al., 2026). For each pair of leaves, a
    homoplasy at a character is a shared, non-missing edit whose lowest common
    ancestor (LCA) is still in the unmodified state, so the edit must have arisen
    independently on the two lineages. Observing more homoplasies than the model
    expects is evidence of an incorrect reconstruction; the cPHS is the minimum
    Benjamini-Hochberg-adjusted p-value across all leaf pairs.

    Ancestral character states must already be annotated on every node under the
    ``characters_key`` node attribute; call
    :func:`cassiopeia.tl.ancestral_characters` first if they are not. The tree
    must be ultrametric (all leaves at equal depth under ``time_key``); use
    :func:`cassiopeia.tl.rescale_node_times` to normalize node times if needed.

    .. warning::

        **Ancestral states must be inferred with parsimony.** A homoplasy is
        counted only when the two leaves' most recent common ancestor is
        still unmutated, so every internal node needs a character state.
        Zilber et al. (2026) show that under irreversible (CRISPR-Cas9)
        editing, parsimony is the appropriate choice: a parent's state is
        determined by its children without ambiguity in all but one case,
        where two siblings carry the same edit and the parent is assigned
        that edit. :func:`cassiopeia.tl.ancestral_characters` implements
        this. Substituting a different ancestral reconstruction procedure
        will change the score and is not recommended. Note also that states
        inferred on an incorrect topology are themselves incorrect, so the
        score reflects the reconstruction as a whole.

        **Do not compare raw scores across trees.** The score is a p-value
        whose scale depends on the number of leaves, the number of
        characters, and the estimated mutation rate and collision
        probability. To compare tree-reconstruction algorithms, compare the
        pass/fail calls obtained by thresholding the score (a tree passes if
        cPHS >= t), using the same ancestral reconstruction for every tree.

        **Choosing a threshold.** The thresholds below were calibrated in
        Zilber et al. (2026) on simulated trees with known ground truth,
        generated by a birth-death process with a CRISPR-Cas9 mutation
        overlay, over n in {100, 200, 300, 500, 1000} leaves, k in {10, 15,
        20, 25, 30} characters, mutation probability rho in {0.5, 0.7, 0.9,
        0.99} and collision probability q in {0.02, 0.1, 0.33}, with 100
        repetitions per configuration. A reconstruction was labelled
        accurate if its normalized Robinson-Foulds distance to the truth was
        below 0.5, and the threshold maximizing balanced accuracy was
        selected.

        =======  ==============  ==============  ==========
        k        rho = 0.5-0.6   rho = 0.7-0.9   rho = 0.99
        =======  ==============  ==============  ==========
        16-24    0.05            1e-3            1e-4
        >= 25    1e-3            1e-4            1e-4
        =======  ==============  ==============  ==========

        Outside this regime the calibration does not apply. For k <= 15
        there are too few recording sites to provide enough data, and no
        threshold separated accurate from inaccurate trees, so the test is
        not recommended in this case. For collision probability above
        roughly 0.35 the scores shift upward and become unreliable, so the
        test is less suitable. For trees much larger than n = 1000 the
        calibration simulations were not extended systematically, as this
        range was not required for the biological dataset analysed; limited
        simulations at k <= 30 suggest that larger trees need more
        characters, so a dedicated calibration is recommended when applying
        the test at n > 1000.

    Args:
        tdata: TreeData object to operate on.
        characters_key: Node attribute holding character states (leaves and,
            after :func:`cassiopeia.tl.ancestral_characters`, internal nodes).
            Also the ``obsm`` key of the character matrix used to estimate the
            mutation rate and collision probability.
        time_key: Node attribute holding node times/depths. Leaves must share a
            common depth; heights are normalized to ``[0, 1]``.
        mutation_rate: Mutation rate ``lambda`` of the model. Estimated from the
            fraction of mutated entries when ``None``.
        collision_probability: Probability ``q`` that two independent edits
            produce the same state. Estimated from ``priors`` when ``None``.
        priors: Prior state probabilities used to estimate ``collision_probability``,
            as a mapping of character index to per-state dicts or a flat
            state->probability dict. Resolved from ``tdata.uns['priors']`` when
            ``None``.
        missing_state: Missing-data value. Resolved from ``tdata`` when ``None``.
        unmodified_state: Unmodified (uncut) state value. Resolved from ``tdata``
            when ``None``.
        tree_key: The ``obst`` key of the tree to use. When ``None`` and multiple
            trees are present, the cPHS is computed for every tree and returned
            as a dictionary keyed by tree name.

    Returns:
        The cPHS score as a float, or, when ``tree_key`` is ``None`` and ``tdata``
        contains multiple trees, a dictionary mapping each tree name to its score.

    Raises:
        TreeMetricError: If a node is missing character states, or the tree is not
            ultrametric under ``time_key``.
    """
    missing_state = _resolve_missing_state(tdata, missing_state)
    unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    if isinstance(unmodified_state, (list, tuple, set)):
        unmodified_state = next(iter(unmodified_state))

    character_matrix = _get_characters(tdata, characters_key)

    # Mutation rate and collision probability are properties of the character
    # matrix / priors and so are estimated once, independent of the tree.
    if mutation_rate is None:
        proportion_mutated = parameter_estimators.fraction_mutated(
            tdata,
            characters_key=characters_key,
            missing_state=missing_state,
            unmodified_state=unmodified_state,
            key_added=None,
        )
        mutation_rate = -np.log(1.0 - proportion_mutated)

    if collision_probability is None:
        priors = _get_parameter(tdata, "priors", value=priors)
        collision_probability = _collision_probability(
            priors, character_matrix, missing_state, unmodified_state
        )

    # Resolve which trees to score. Return a dict only when scoring every tree of
    # a multi-tree TreeData; a single explicit tree_key always yields a scalar.
    if tree_key is not None:
        tree_keys = [tree_key]
        return_dict = False
    else:
        tree_keys = list(tdata.obst.keys())
        return_dict = len(tree_keys) > 1

    scores = {}
    for key in tree_keys:
        g, key = _get_digraph(tdata, key)
        for node in g.nodes:
            if characters_key not in g.nodes[node]:
                raise TreeMetricError(
                    f"Node {node!r} has no character states under {characters_key!r}. "
                    "Call cassiopeia.tl.ancestral_characters first. Note that cPHS "
                    "is strongly influenced by ancestral reconstruction accuracy."
                )
        scores[key] = _calculate_cphs(
            g,
            mutation_rate,
            collision_probability,
            time_key,
            characters_key,
            missing_state,
            unmodified_state,
        )

    return scores if return_dict else scores[tree_keys[0]]
