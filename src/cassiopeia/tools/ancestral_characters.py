import networkx as nx
import numpy as np

from cassiopeia.mixins import is_ambiguous_state


def _get_lca_characters(
    vecs: list[list[int] | list[tuple[int, ...]]],
    missing_state_indicator: int,
) -> list[int]:
    """Builds the character vector of the LCA of a list of character vectors, obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Importantly, this method will infer ancestral characters for an ambiguous
    state. If the intersection between two states (even ambiguous) is non-zero
    and not the missing state, and has length exactly 1, we assign the ancestral
    state this value. Else, if the intersection length is greater than 1, the
    value '0' is assigned.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_state_indicator: The character representing missing values

    Returns:
            A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [0] * len(vecs[0])
    for i in range(k):
        if np.all(np.array([vec[i] for vec in vecs], dtype=object) == missing_state_indicator):
            lca_vec[i] = missing_state_indicator
        else:
            all_states = [vec[i] for vec in vecs if vec[i] != missing_state_indicator]

            # this check is specifically if all_states consists of a single
            # ambiguous state.
            if len(list(set(all_states))) == 1:
                state = all_states[0]
                # lca_vec[i] = state
                if is_ambiguous_state(state) and len(state) == 1:
                    lca_vec[i] = state[0]
                else:
                    lca_vec[i] = all_states[0]
            else:
                all_ambiguous = np.all([is_ambiguous_state(s) for s in all_states])
                chars = set.intersection(
                    *map(
                        set,
                        [state if is_ambiguous_state(state) else [state] for state in all_states],
                    )
                )
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
                if all_ambiguous:
                    # if we only have ambiguous states, we set the LCA state
                    # to be the intersection.
                    lca_vec[i] = tuple(chars)
    return lca_vec


def reconstruct_ancestral_characters(
    g, missing_state_indicator=-1, key_added="character_states"
) -> None:
    """Reconstruct ancestral character states.

    Reconstructs ancestral states (i.e., those character states in the
    internal nodes) using the Camin-Sokal parsimony criterion (i.e.,
    irreversibility). Operates on the tree in place.

    Raises:
        AttributeError if the tree has not been initialized.
    """
    for n in nx.dfs_postorder_nodes(g):
        if g.out_degree(n) == 0:
            if g.nodes[n][key_added][:] == []:
                raise AttributeError(
                    "Character states have not been initialized at leaves."
                    " Use set_character_states_at_leaves or populate_tree"
                    " with the character matrix that specifies the leaf"
                    " character states."
                )
            continue
        children = g.successors(n)
        character_states = [g.nodes[c][key_added][:] for c in children]
        reconstructed = _get_lca_characters(character_states, missing_state_indicator)
        g.nodes[n][key_added] = reconstructed
