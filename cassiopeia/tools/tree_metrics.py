"""
This file stores functions for scoring metrics on a tree.
Currently, we'll support parsimony and likelihood calculations.
"""
from typing import Optional, Union

import numpy as np
import scipy

from cassiopeia.data.Layers import Layers
from cassiopeia.mixins import CassiopeiaTreeError
from cassiopeia.data import CassiopeiaTree
import cassiopeia.tools.parameter_estimators as pe


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
    nodes' character states are inferred by Camin-Sokal Parsimony.
    Otherwise, the existing annotations at the internal states are used.
    If `treat_missing_as_mutations` is set to True, then transitions
    from a non-missing state to a missing state are counted in the
    parsimony calculation. Otherwise, they are not included.

    Args:
        tree: The tree to calculate parsimony over
        infer_ancestral_characters: Whether to infer the ancestral
            characters states of the tree
        treat_missing_as_mutations: Whether to treat missing states as
            mutations

    Returns:
        The number of mutations that have occurred on the tree

    Raises:
        CassiopeiaTreeError if the tree has not been initialized or if
            a node does not have character states initialized

    """
    tree._CassiopeiaTree__check_network_initialized()

    if infer_ancestral_characters:
        tree.reconstruct_ancestral_characters()

    parsimony = 0

    if tree.get_character_states(tree.root) == []:
        raise CassiopeiaTreeError(
            f"Character states empty at internal node. Annotate"
            " character states or infer ancestral characters by"
            " setting infer_ancestral_characters=True."
        )

    for u, v in tree.depth_first_traverse_edges():
        if tree.get_character_states(v) == []:
            if tree.is_leaf(v):
                raise CassiopeiaTreeError(
                    "Character states have not been initialized at leaves."
                    " Use set_character_states_at_leaves or populate_tree"
                    " with the character matrix that specifies the leaf"
                    " character states."
                )
            else:
                raise CassiopeiaTreeError(
                    f"Character states empty at internal node. Annotate"
                    " character states or infer ancestral characters by"
                    " setting infer_ancestral_characters=True."
                )

        parsimony += len(
            tree.get_mutations_along_edge(u, v, treat_missing_as_mutation)
        )

    return parsimony


def calculate_likelihood(
    tree: CassiopeiaTree,
    use_branch_lengths: bool = True,
    use_internal_character_states: bool = False,
    proportion_of_missing_as_stochastic: float = 0.5,
    layer: Optional[str] = None,
):
    """
    Calculates the log likelihood of a tree with irreversible mutations.

    Calculates the log likelihood of a tree given the character states at
    the leaves using Felsenstein's Pruning Algorithm, which sets up a
    recursive relation between the likelihoods of states at nodes. The
    likelihood L(s, n) at a given state s at a given node n is:

    L(s, n) = Π_{n'}(Σ_{s'}(P(s'|s) * L(s', n')))

    for all n' that are children of n, and s' in the state space, with
    P(s'|s) being the transition probability from s to s'. That is,
    the likelihood at a given state at a given node is the product of
    the likelihoods of the states at the children scaled by the
    probability of the current state transitioning to those states.

    We assume here that characters mutate independently and that mutations
    are irreversible. Once a character mutates to a certain state that
    character cannot mutate again. To determine the probability of
    acquiring a given state once a mutation occurs, the priors of the
    tree are used.

    The likelihood depends on rates for mutation, heritable missing data, and
    stochastic missing data. For the first two, if branch lengths are to be
    used, these rates are per-generation rates, i.e. the probability that a
    mutation/missing data occurs on a branch. If branch lengths are to be
    used, this rate is the instantaneous rate assuming that the waiting time
    until a mutation/missing data is exponentially distributed. The
    probability that an event occurred in time t is then given by the
    exponential CDF. The rate for stochastic missing data is a flat rate and
    represents the probability at which a stochastic missing event will
    occur at a character on a leaf, given that no heritable missing data has
    already occurred on that character.

    This function attempts to consume these rates from `tree.parameters` as
    `mutation_rate`, `heritable_missing_rate`, and
    `stochastic_missing_probability`. If the rates are not found, then they
    are estimated. As heritable and stochastic missing data are convolved in
    the total missing data, one can be estimated from the other. If
    neither are provide, the user must provide the proportion of the total
    missing data that is believed to be due to stochastic missing data events
    (specified by `proportion_of_missing_as_stochastic`), and the heritable
    missing data is then estimated using the remaining proportion of missing
    data that is then heritable. The default is 0.5.

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
        tree: The tree on which to calculate likelihood over
        use_branch_lengths: Indicates if branch lengths should be taken into
            account in the likelihood calculation
        use_internal_character_states: Indicates if internal node
            character states should be assumed to be specified exactly
        mutation_rate: The rate at which mutations occur on a given lineage
        heritable_missing_rate: The rate at which heritable missing data occurs
            on a given lineage
        stochastic_missing_probability: The probability that a character/
            state pair acquires stochastic missing data
        proportion_of_missing_as_stochastic: The assumed proportion of
            missing data that is stochastic if neither missing data rates are
            explicitly provided
        layer: Layer to use for the character matrix in estimating parameters.
            If this is None, then the current `character_matrix` variable will
            be used.

    Returns:
        The log likelihood of the tree given the observed character states.

    Raises:
        CassiopeiaError if the parameters consumed from the tree are invalid,
            if the tree priors are not populated, or if character states 
            annotations are missing at a node.
    """

    def log_transition_probability(
        s: Union[int, str], s_: Union[int, str], t: float, character: int
    ) -> float:
        """Gives the log transition probability between two given states.

        Gives the transition probability between two states assuming
        irreversibility and that 0 is the uncut state. Uses the priors on the
        tree to represent the probability of acquiring a non-0 state.

        Args:
            s: The original state
            s_: The state being transitioned to
            t: The length of time that the transition can occur along
            character: The character whose distribution to draw the prior from

        Returns:
            The log transition probability between the states
        """
        if s_ == tree.missing_state_indicator:
            if s == tree.missing_state_indicator:
                return 0
            else:
                return np.log(
                    missing_probability_function_of_time(
                        heritable_missing_rate, t
                    )
                )
        # "&" represents all non-missing states (including the uncut state).
        # The sum probability of transitioning from any non-missing state s
        # to any non-missing state s' is 1 - P(missing event). Used to avoid
        # marginalizing over the entire state space.
        elif s_ == "&":
            if s == tree.missing_state_indicator:
                return -1e16
            else:
                return np.log(
                    1
                    - missing_probability_function_of_time(
                        heritable_missing_rate, t
                    )
                )
        elif s_ == 0:
            if s != 0 or s == tree.missing_state_indicator:
                return -1e16
            elif s == 0:
                return np.log(
                    1 - mutation_probability_function_of_time(mutation_rate, t)
                ) + np.log(
                    1
                    - missing_probability_function_of_time(
                        heritable_missing_rate, t
                    )
                )
            else:
                # In practice, the transition from "&" to the uncut state 
                # does not occur
                return np.log(
                    1
                    - missing_probability_function_of_time(
                        heritable_missing_rate, t
                    )
                )
        else:
            if s == tree.missing_state_indicator:
                return -1e16
            elif s == 0:
                return (
                    np.log(
                        mutation_probability_function_of_time(mutation_rate, t)
                    )
                    + np.log(tree.priors[character][s_])
                    + np.log(
                        1
                        - missing_probability_function_of_time(
                            heritable_missing_rate, t
                        )
                    )
                )
            elif s == s_:
                return np.log(
                    1
                    - missing_probability_function_of_time(
                        heritable_missing_rate, t
                    )
                )
            else:
                # In practice, the transition from "&" to a non-uncut state 
                # does not occur
                return -1e16

    def log_likelihood_of_character(ind: int) -> float:
        """Calculates the log likelihood of a given character on the tree.

        Args:
            ind: The index of the character to calculate the likelihood of

        Returns:
            The log likelihood of the tree on one character
        """

        # We track the likelihoods for each state at each node
        likelihoods_at_nodes = {}

        # Perform a DFS to propagate the likelihood from the leaves
        for n in tree.depth_first_traverse_nodes(postorder=True):
            state_at_n = tree.get_character_states(n)
            # If states are observed, their likelihoods are set to 1
            if tree.is_leaf(n):
                likelihoods_at_nodes[n] = {state_at_n[ind]: 0}
                continue

            possible_states = []
            # If internal character states are to be used, then the likelihood
            # for all other states are ignored. Otherwise, marginalize over
            # only states that do not break irreversibility, as all states that
            # do have likelihood of 0
            if use_internal_character_states:
                possible_states = [state_at_n[ind]]
            else:
                child_possible_states = []
                for c in [
                    set(likelihoods_at_nodes[child])
                    for child in tree.children(n)
                ]:
                    if tree.missing_state_indicator not in c and "&" not in c:
                        child_possible_states.append(c)
                # '*' represents all non-missing states (including uncut), and
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
                    possible_states = list(
                        set.intersection(*child_possible_states)
                    )
                    if 0 not in possible_states:
                        possible_states.append(0)

            likelihoods_at_n = {}

            # We calculate the likelihood of the states at
            # the current node according to the recurrence relation
            for s in possible_states:
                likelihood_for_s = 0
                for child in tree.children(n):
                    likelihoods_for_s_marginalize_over_s_ = []
                    for s_ in likelihoods_at_nodes[child]:
                        likelihood_s_ = (
                            log_transition_probability(
                                s,
                                s_,
                                tree.get_branch_length(n, child),
                                ind,
                            )
                            + likelihoods_at_nodes[child][s_]
                        )
                        # Here we take into account the probability of
                        # stochastic missing data
                        if tree.is_leaf(child):
                            if (
                                s_ == tree.missing_state_indicator
                                and s != tree.missing_state_indicator
                            ):
                                likelihood_s_ = np.log(
                                    np.exp(likelihood_s_)
                                    + (
                                        1
                                        - missing_probability_function_of_time(
                                            heritable_missing_rate,
                                            tree.get_branch_length(n, child),
                                        )
                                    )
                                    * stochastic_missing_probability
                                )
                            if s_ != tree.missing_state_indicator:
                                likelihood_s_ += np.log(
                                    1 - stochastic_missing_probability
                                )
                        likelihoods_for_s_marginalize_over_s_.append(
                            likelihood_s_
                        )
                    likelihood_for_s += scipy.special.logsumexp(
                        np.array(likelihoods_for_s_marginalize_over_s_)
                    )
                likelihoods_at_n[s] = likelihood_for_s

            likelihoods_at_nodes[n] = likelihoods_at_n

        # If we are not to use the internal state annotations explicitly,
        # then we impose an implicit root of all 0s if it does not exist
        if (
            not use_internal_character_states
            and len(tree.children(tree.root)) != 1
        ):
            # If branch lengths are to be used, we calculate the length of the
            # implicit root edge as the average of all other edges
            likelihood_contribution_from_each_root_state = [
                log_transition_probability(
                    0, s_, implicit_root_branch_length, ind
                )
                + likelihoods_at_nodes[tree.root][s_]
                for s_ in likelihoods_at_nodes[tree.root]
            ]
            likelihood_at_implicit_root = scipy.special.logsumexp(
                likelihood_contribution_from_each_root_state
            )

            return likelihood_at_implicit_root

        else:
            return list(likelihoods_at_nodes[tree.root].values())[0]

    if tree.priors is None:
        raise CassiopeiaTreeError(
            "Priors must be specified for this tree to calculate the"
            " likelihood."
        )

    for l in tree.leaves:
        if tree.get_character_states(l) == []:
            raise CassiopeiaTreeError(
                "Character states have not been initialized at leaves."
                " Use set_character_states_at_leaves or populate_tree"
                " with the character matrix that specifies the leaf"
                " character states."
            )

    if use_internal_character_states:
        for i in tree.internal_nodes:
            if tree.get_character_states(i) == []:
                raise CassiopeiaTreeError(
                    "Character states empty at internal node. Character"
                    " states must be annotated at each node if internal"
                    " character states are to be used."
                )
    if (
        proportion_of_missing_as_stochastic < 0
        or proportion_of_missing_as_stochastic > 1
    ):
        raise CassiopeiaTreeError(
            "Proportion of missing as stochastic must be between 0 and 1"
        )

    # Here we attempt to consume the lineage tracing parameters from the tree.
    # If the attributes are not populated, then the parameters are inferred.
    if "mutation_rate" not in tree.parameters:
        mutation_rate = pe.estimate_mutation_rate(
            tree, use_branch_lengths, (not use_internal_character_states), layer
        )
    else:
        mutation_rate = tree.parameters["mutation_rate"]
    if "stochastic_missing_probability" not in tree.parameters:
        stochastic_missing_probability = (
            pe.estimate_stochastic_missing_data_probability(
                tree,
                use_branch_lengths,
                (not use_internal_character_states),
                proportion_of_missing_as_stochastic,
                layer,
            )
        )
    else:
        stochastic_missing_probability = tree.parameters[
            "stochastic_missing_probability"
        ]
    if "heritable_missing_rate" not in tree.parameters:
        heritable_missing_rate = pe.estimate_heritable_missing_data_rate(
            tree,
            use_branch_lengths,
            (not use_internal_character_states),
            proportion_of_missing_as_stochastic,
            layer,
        )
    else:
        heritable_missing_rate = tree.parameters["heritable_missing_rate"]

    # We check that the mutation and missing rates have valid values
    if mutation_rate < 0:
        raise CassiopeiaTreeError("Mutation rate must be > 0.")
    if not use_branch_lengths and mutation_rate > 1:
        raise CassiopeiaTreeError("Per-generation mutation rate must be < 1.")
    if heritable_missing_rate < 0:
        raise CassiopeiaTreeError("Heritable missing data rate must be > 0.")
    if not use_branch_lengths and heritable_missing_rate > 1:
        raise CassiopeiaTreeError(
            "Per-generation heritable missing data rate must be < 1."
        )
    if stochastic_missing_probability < 0:
        raise CassiopeiaTreeError("Stochastic missing data rate must be > 0.")
    if stochastic_missing_probability > 1:
        raise CassiopeiaTreeError("Stochastic missing data rate must be < 1.")

    if not use_branch_lengths:
        mutation_probability_function_of_time = lambda rate, t: rate
        missing_probability_function_of_time = lambda rate, t: rate
        implicit_root_branch_length = 1

    else:
        mutation_probability_function_of_time = lambda rate, t: 1 - np.exp(
            -rate * t
        )
        missing_probability_function_of_time = lambda rate, t: 1 - np.exp(
            -rate * t
        )
        implicit_root_branch_length = np.mean(
            [tree.get_branch_length(u, v) for u, v in tree.edges]
        )

    return np.sum(
        [log_likelihood_of_character(char) for char in range(tree.n_character)]
    )
