"""File containing functions for scoring metrics on a tree."""

from collections.abc import Callable

import numpy as np
import scipy

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import TreeMetricError
from cassiopeia.tools import parameter_estimators


def calculate_parsimony(
    tree: CassiopeiaTree,
    infer_ancestral_characters: bool = False,
    treat_missing_as_mutation: bool = False,
) -> int:
    """
    Calculates the number of mutations that have occurred on a tree.

    Calculates the parsimony, defined as the number of character/state
    mutations that occur on edges of the tree, from the character state
    annotations at the nodes. A mutation is said to have occurred on an
    edge if a state is present at a character at the child node and this
    state is not in the parent node.

    If `infer_ancestral_characters` is set to True, then the internal
    nodes' character states are inferred by Camin-Sokal Parsimony from the
    current character states at the leaves. Use
    `tree.set_character_states_at_leaves` to use a different layer to infer
    ancestral states. Otherwise, the current annotations at the internal
    states are used. If `treat_missing_as_mutations` is set to True, then
    transitions from a non-missing state to a missing state are counted in
    the parsimony calculation. Otherwise, they are not included.

    Args:
        tree: The tree to calculate parsimony over
        infer_ancestral_characters: Whether to infer the ancestral
            characters states of the tree
        treat_missing_as_mutations: Whether to treat missing states as
            mutations

    Returns
    -------
        The number of mutations that have occurred on the tree

    Raises
    ------
        TreeMetricError if the tree has not been initialized or if
            a node does not have character states initialized
    """
    if infer_ancestral_characters:
        tree.reconstruct_ancestral_characters()

    parsimony = 0

    if tree.get_character_states(tree.root) == []:
        raise TreeMetricError(
            "Character states empty at internal node. Annotate"
            " character states or infer ancestral characters by"
            " setting infer_ancestral_characters=True."
        )

    for u, v in tree.depth_first_traverse_edges():
        if tree.get_character_states(v) == []:
            if tree.is_leaf(v):
                raise TreeMetricError(
                    "Character states have not been initialized at leaves."
                    " Use set_character_states_at_leaves or populate_tree"
                    " with the character matrix that specifies the leaf"
                    " character states."
                )
            else:
                raise TreeMetricError(
                    "Character states empty at internal node. Annotate"
                    " character states or infer ancestral characters by"
                    " setting infer_ancestral_characters=True."
                )

        parsimony += len(tree.get_mutations_along_edge(u, v, treat_missing_as_mutation))

    return parsimony


def log_transition_probability(
    tree: CassiopeiaTree,
    character: int,
    s: int | str,
    s_: int | str,
    t: float,
    mutation_probability_function_of_time: Callable[[float], float],
    missing_probability_function_of_time: Callable[[float], float],
) -> float:
    """Gives the log transition probability between two given states.

    Assumes that 0 is the uncut-state, and that only 0 has non-zero probability
    of transitioning to non-0 states. Additionally, assumes that any non-missing
    state can mutate to the missing state, specified by
    `tree.missing_state_indicator`.

    Here, "&" represents a placeholder for any non-missing state. Thus, the
    probability of transitioning from any state to any given non-missing
    state is (1 - probability of transitioning to the missing state).

    The probability of acquiring a mutation is given by a time t and
    `mutation_probability_function_of_time`, and the same is true of heritable
    missing data events and `missing_probability_function_of_time`. In
    determining the probability of acquiring a non-missing, non-0 state, the
    priors on the tree (`tree.priors`) are used.

    Args:
        tree: The tree on which the priors are stored
        character: The character whose distribution to draw the prior from
        s: The original state
        s_: The state being transitioned to
        t: The length of time that the transition can occur along
        mutation_probability_function_of_time: The function defining the
            probability of a lineage acquiring a mutation within a given time
        missing_probability_function_of_time: The function defining the
            probability of a lineage acquiring heritable missing data within a
            given time

    Returns
    -------
        The log transition probability between the states
    """
    if s_ == tree.missing_state_indicator:
        if s == tree.missing_state_indicator:
            return 0
        else:
            return np.log(missing_probability_function_of_time(t))
    # "&" stands in for any non-missing state (including the uncut state).
    # The sum probability of transitioning from any non-missing state s
    # to any non-missing state s' is 1 - P(missing event). Used to avoid
    # marginalizing over the entire state space.
    elif s_ == "&":
        if s == tree.missing_state_indicator:
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
        if s == tree.missing_state_indicator:
            return -1e16
        elif s == 0:
            return (
                np.log(mutation_probability_function_of_time(t))
                + np.log(tree.priors[character][s_])
                + np.log(1 - missing_probability_function_of_time(t))
            )
        elif s == s_:
            return np.log(1 - missing_probability_function_of_time(t))
        else:
            # The transition from "&" to a non-missing state cannot occur
            return -1e16


def log_likelihood_of_character(
    tree: CassiopeiaTree,
    character: int,
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
    P(s'|s) being the transition probability from s to s'. That is,
    the likelihood at a given state at a given node is the product of
    the likelihoods of the states at this character at the children scaled by
    the probability of the current state transitioning to those states. This
    includes the missing state, as specified by `tree.missing_state_indicator`.

    We assume here that mutations are irreversible. Once a character mutates to
    a certain state that character cannot mutate again, with the exception of
    the fact that any non-missing state can mutate to a missing state.
    `mutation_probability_function_of_time` is expected to be a function that
    determine the probability of a mutation occuring given an amount of time.
    To determine the probability of acquiring a given (non-missing) state once
    a mutation occurs, the priors of the tree are used. Likewise,
    `missing_probability_function_of_time` determines the the probability of a
    missing data event occuring given an amount of time.

    The user can choose to use the character states annotated at internal
    nodes. If these are not used, then the likelihood is marginalized over
    all possible internal state characters. If the actual internal states
    are not provided, then the root is assumed to have the unmutated state
    at each character. Additionally, it is assumed that there is a single
    branch leading from the root that represents the roots' lifetime. If
    this branch does not exist and `use_internal_character_states` is set
    to False, then this branch is added with branch length equal to the
    average branch length of this tree.

    Args:
        tree: The tree on which to calculate the likelihood
        character: The index of the character to calculate the likelihood of
        use_internal_character_states: Indicates if internal node
            character states should be assumed to be specified exactly
        mutation_probability_function_of_time: The function defining the
            probability of a lineage acquiring a mutation within a given time
        missing_probability_function_of_time: The function defining the
            probability of a lineage acquiring heritable missing data within a
            given time
        stochastic_missing_probability: The probability that a cell/character
            pair acquires stochastic missing data at the end of the lineage
        implicit_root_branch_length: The length of the implicit root branch.
            Used if the implicit root needs to be added

    Returns
    -------
        The log likelihood of the tree on one character
    """
    # This dictionary uses a nested dictionary structure. Each node is mapped
    # to a dictionary storing the likelihood for each possible state
    # (states that have non-0 likelihood)
    likelihoods_at_nodes = {}

    # Perform a DFS to propagate the likelihood from the leaves
    for n in tree.depth_first_traverse_nodes(postorder=True):
        state_at_n = tree.get_character_states(n)
        # If states are observed, their likelihoods are set to 1
        if tree.is_leaf(n):
            likelihoods_at_nodes[n] = {state_at_n[character]: 0}
            continue

        possible_states = []
        # If internal character states are to be used, then the likelihood
        # for all other states are ignored. Otherwise, marginalize over
        # only states that do not break irreversibility, as all states that
        # do have likelihood of 0
        if use_internal_character_states:
            possible_states = [state_at_n[character]]
        else:
            child_possible_states = []
            for c in [set(likelihoods_at_nodes[child]) for child in tree.children(n)]:
                if tree.missing_state_indicator not in c and "&" not in c:
                    child_possible_states.append(c)
            # "&" stands in for any non-missing state (including uncut), and
            # is a possible state when all children are missing, as any
            # state could have occurred at the parent if all missing data
            # events occurred independently. Used to avoid marginalizing
            # over the entire state space.
            if child_possible_states == []:
                possible_states = [
                    "&",
                    tree.missing_state_indicator,
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
            for child in tree.children(n):
                likelihoods_for_s_marginalize_over_s_ = []
                for s_ in likelihoods_at_nodes[child]:
                    likelihood_s_ = (
                        log_transition_probability(
                            tree,
                            character,
                            s,
                            s_,
                            tree.get_branch_length(n, child),
                            mutation_probability_function_of_time,
                            missing_probability_function_of_time,
                        )
                        + likelihoods_at_nodes[child][s_]
                    )
                    # Here we take into account the probability of
                    # stochastic missing data
                    if tree.is_leaf(child):
                        if s_ == tree.missing_state_indicator and s != tree.missing_state_indicator:
                            likelihood_s_ = np.log(
                                np.exp(likelihood_s_)
                                + (1 - missing_probability_function_of_time(tree.get_branch_length(n, child)))
                                * stochastic_missing_probability
                            )
                        if s_ != tree.missing_state_indicator:
                            likelihood_s_ += np.log(1 - stochastic_missing_probability)
                    likelihoods_for_s_marginalize_over_s_.append(likelihood_s_)
                likelihood_for_s += scipy.special.logsumexp(np.array(likelihoods_for_s_marginalize_over_s_))
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
        if len(tree.children(tree.root)) != 1:
            likelihood_contribution_from_each_root_state = [
                log_transition_probability(
                    tree,
                    character,
                    0,
                    s_,
                    implicit_root_branch_length,
                    mutation_probability_function_of_time,
                    missing_probability_function_of_time,
                )
                + likelihoods_at_nodes[tree.root][s_]
                for s_ in likelihoods_at_nodes[tree.root]
            ]
            likelihood_at_implicit_root = scipy.special.logsumexp(likelihood_contribution_from_each_root_state)

            return likelihood_at_implicit_root

        else:
            # Here we account for the edge case in which all of the leaves are
            # missing, in which case the root will have "&" in place of 0. The
            # likelihood at "&" will have the same likelihood as 0 based on the
            # transition rules regarding "&". As "&" is a placeholder when the
            # state is unknown, this can be thought of realizing "&" as 0.
            if 0 not in likelihoods_at_nodes[tree.root]:
                return likelihoods_at_nodes[tree.root]["&"]
            else:
                # Otherwise, we return the likelihood of the 0 state at the
                # existing implicit root
                return likelihoods_at_nodes[tree.root][0]

    # If we use the internal state annotations explicitly, then we return
    # the likelihood of the state annotated at this character at the root
    else:
        return list(likelihoods_at_nodes[tree.root].values())[0]


def get_lineage_tracing_parameters(
    tree: CassiopeiaTree,
    continuous: bool,
    assume_root_implicit_branch: bool,
    layer: str | None = None,
) -> tuple[float, float, float]:
    """Gets the lineage tracing parameters from a tree.

    This function attempts to consume these parameters from `tree.parameters`
    as `mutation_rate`, `heritable_missing_rate`, and
    `stochastic_missing_probability`. If the rates are not found, then they
    are estimated using their respective estimators. Note that in order to
    estimate the missing data parameters, at least one must be populated in
    `tree.parameters`

    Args:
        tree: The tree on which to consume/estimate the parameters
        continuous: If the parameters are to be estimated, then whether to
            estimate them as continuous or discrete parameters
        assume_root_implicit_branch: In the tree depth/time in estimating the
            rate parameters, whether or not to include an implicit root
        layer: Layer to use for the character matrix in estimating parameters.
            If this is None, then the current `character_matrix` variable will
            be used.

    Returns
    -------
        The mutation rate, the heritable missing rate, and the stochastic
            missing probability

    Raises
    ------
        TreeMetricError if one of the provided parameters is invalid
    """
    # Here we attempt to consume the lineage tracing parameters from the tree.
    # If the attributes are not populated, then the parameters are inferred.
    if "mutation_rate" not in tree.parameters:
        mutation_rate = parameter_estimators.estimate_mutation_rate(
            tree, continuous, assume_root_implicit_branch, layer
        )
    else:
        mutation_rate = tree.parameters["mutation_rate"]
    if not ("stochastic_missing_probability" in tree.parameters and "heritable_missing_rate" in tree.parameters):
        (
            stochastic_missing_probability,
            heritable_missing_rate,
        ) = parameter_estimators.estimate_missing_data_rates(
            tree,
            continuous,
            assume_root_implicit_branch,
            layer=layer,
        )
    else:
        stochastic_missing_probability = tree.parameters["stochastic_missing_probability"]
        heritable_missing_rate = tree.parameters["heritable_missing_rate"]

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


def calculate_likelihood_discrete(
    tree: CassiopeiaTree,
    use_internal_character_states: bool = False,
    layer: str | None = None,
) -> float:
    """
    Calculates the log likelihood of a tree under a discrete process.

    A wrapper function for `get_lineage_tracing_parameters` and
    `log_likelihood_of_character` under a discrete model of lineage tracing.

    This function acquires the mutation rate, the heritable missing rate, and
    the stochastic missing probability from the tree using
    `get_lineage_tracing_parameters`. The rates are assumed to be per-generation
    rates. Then, it calculates the log likelihood for each character using
    `log_likelihood_of_character`, and then by assumption that characters
    mutate independently, sums their likelihoods to get the likelihood for the
    tree.

    Here, branch lengths are not to be used. We assume that rates are
    per-generation, that is, these rates represent the probability a mutation
    occurs on a branch regardless of the branch length t.

    Args:
        tree: The tree on which to calculate likelihood over
        use_internal_character_states: Indicates if internal node
            character states should be assumed to be specified exactly
        layer: Layer to use for the character matrix in estimating parameters.
            If this is None, then the current `character_matrix` variable will
            be used.

    Returns
    -------
        The log likelihood of the tree given the observed character states.

    Raises
    ------
        CassiopeiaError if the parameters consumed from the tree are invalid,
            if the tree priors are not populated, or if character states
            annotations are missing at a node.
    """
    if tree.priors is None:
        raise TreeMetricError("Priors must be specified for this tree to calculate the likelihood.")

    for l in tree.leaves:
        if tree.get_character_states(l) == []:
            raise TreeMetricError(
                "Character states have not been initialized at leaves."
                " Use set_character_states_at_leaves or populate_tree"
                " with the character matrix that specifies the leaf"
                " character states."
            )

    if use_internal_character_states:
        for i in tree.internal_nodes:
            if tree.get_character_states(i) == []:
                raise TreeMetricError(
                    "Character states empty at internal node. Character"
                    " states must be annotated at each node if internal"
                    " character states are to be used."
                )

    (
        mutation_rate,
        heritable_missing_rate,
        stochastic_missing_probability,
    ) = get_lineage_tracing_parameters(
        tree,
        False,
        (not use_internal_character_states),
        layer,
    )

    mutation_probability_function_of_time = lambda t: mutation_rate
    missing_probability_function_of_time = lambda t: heritable_missing_rate

    return np.sum(
        [
            log_likelihood_of_character(
                tree,
                character,
                use_internal_character_states,
                mutation_probability_function_of_time,
                missing_probability_function_of_time,
                stochastic_missing_probability,
                implicit_root_branch_length=1,
            )
            for character in range(tree.n_character)
        ]
    )


def calculate_likelihood_continuous(
    tree: CassiopeiaTree,
    use_internal_character_states: bool = False,
    layer: str | None = None,
) -> float:
    """
    Calculates the log likelihood of a tree under a continuous process.

    A wrapper function for `get_lineage_tracing_parameters` and
    `log_likelihood_of_character` under a continuous model of lineage tracing.

    This function acquires the mutation rate, the heritable missing rate, and
    the stochastic missing probability from the tree using
    `get_lineage_tracing_parameters`. The rates are assumed to be instantaneous
    rates. Then, it calculates the log likelihood for each character using
    `log_likelihood_of_character`, and then by assumption that characters
    mutate independently, sums their likelihoods to get the likelihood for the
    tree.

    Here, branch lengths are to be used. We assume that the rates are
    instantaneous rates representing the frequency at which mutation and missing
    data events occur in a period of time. Under this continuous model, we assume
    that the waiting time until a mutation/missing data event is exponentially
    distributed. The probability that an event occurred in time t is thus given
    by the exponential CDF.

    Args:
        tree: The tree on which to calculate likelihood over
        use_internal_character_states: Indicates if internal node
            character states should be assumed to be specified exactly
        layer: Layer to use for the character matrix in estimating parameters.
            If this is None, then the current `character_matrix` variable will
            be used.

    Returns
    -------
        The log likelihood of the tree given the observed character states.

    Raises
    ------
        CassiopeiaError if the tree priors are not populated, or if character
            state annotations are missing at a node.
    """
    if tree.priors is None:
        raise TreeMetricError("Priors must be specified for this tree to calculate the likelihood.")

    for l in tree.leaves:
        if tree.get_character_states(l) == []:
            raise TreeMetricError(
                "Character states have not been initialized at leaves."
                " Use set_character_states_at_leaves or populate_tree"
                " with the character matrix that specifies the leaf"
                " character states."
            )

    if use_internal_character_states:
        for i in tree.internal_nodes:
            if tree.get_character_states(i) == []:
                raise TreeMetricError(
                    "Character states empty at internal node. Character"
                    " states must be annotated at each node if internal"
                    " character states are to be used."
                )

    (
        mutation_rate,
        heritable_missing_rate,
        stochastic_missing_probability,
    ) = get_lineage_tracing_parameters(
        tree,
        True,
        (not use_internal_character_states),
        layer,
    )

    mutation_probability_function_of_time = lambda t: 1 - np.exp(-mutation_rate * t)
    missing_probability_function_of_time = lambda t: 1 - np.exp(-heritable_missing_rate * t)
    implicit_root_branch_length = np.mean([tree.get_branch_length(u, v) for u, v in tree.edges])

    return np.sum(
        [
            log_likelihood_of_character(
                tree,
                character,
                use_internal_character_states,
                mutation_probability_function_of_time,
                missing_probability_function_of_time,
                stochastic_missing_probability,
                implicit_root_branch_length,
            )
            for character in range(tree.n_character)
        ]
    )
