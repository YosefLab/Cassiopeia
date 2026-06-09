from __future__ import annotations

import networkx as nx
import numpy as np
from treedata import TreeData

from cassiopeia.mixins import is_ambiguous_state
from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.utils import (
    _get_characters,
    _get_digraph,
    _get_parameter,
    _normalize_missing,
    _set_tree,
)


def _get_lca_characters(
    vecs: list[list[int] | list[tuple[int, ...]]],
    missing_states: set,
    unmodified_state: int,
) -> list[int]:
    """Builds the character vector of the LCA of a list of character vectors, obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Importantly, this method will infer ancestral characters for an ambiguous
    state. If the intersection between two states (even ambiguous) is non-zero
    and not the missing state, and has length exactly 1, we assign the ancestral
    state this value. Else, if the intersection length is greater than 1, the
    ``unmodified_state`` value is assigned.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_states: The set of values representing missing data
        unmodified_state: The value representing the unmodified (uncut) state

    Returns:
            A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [unmodified_state] * len(vecs[0])
    for i in range(k):
        if all(vec[i] in missing_states for vec in vecs):
            lca_vec[i] = next(iter(missing_states))
        else:
            all_states = [vec[i] for vec in vecs if vec[i] not in missing_states]

            # this check is specifically if all_states consists of a single
            # ambiguous state.
            if len(list(set(all_states))) == 1:
                state = all_states[0]
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


def _seed_leaf_states(g: nx.DiGraph, character_matrix, key_added: str) -> None:
    """Initialize leaf ``key_added`` node attributes from a character matrix.

    Args:
        g: Directed tree graph.
        character_matrix: ``pd.DataFrame`` indexed by leaf name with one column
            per character.
        key_added: Node attribute under which the per-leaf character state list
            is stored.

    Raises:
        CassiopeiaError: If a leaf is missing from *character_matrix*.
    """
    for node in g.nodes:
        if g.out_degree(node) != 0:
            continue
        if node not in character_matrix.index:
            raise CassiopeiaError(
                f"Leaf {node!r} is not present in the character matrix; cannot "
                "seed leaf character states."
            )
        g.nodes[node][key_added] = list(character_matrix.loc[node])


def _reconstruct(g: nx.DiGraph, missing_states: set, unmodified_state: int, key: str) -> None:
    """Reconstruct ancestral states in place using Camin-Sokal parsimony.

    Internal-node states under the ``key`` node attribute are inferred bottom-up
    from the children's states. Leaves are assumed to already carry ``key``.

    Raises:
        AttributeError: If a leaf has no character states under ``key``.
    """
    for n in nx.dfs_postorder_nodes(g):
        if g.out_degree(n) == 0:
            if g.nodes[n].get(key, None) is None or g.nodes[n][key] == []:
                raise AttributeError(
                    "Character states have not been initialized at leaves."
                    " Seed leaf character states from a character matrix"
                    " before reconstructing ancestral characters."
                )
            continue
        character_states = [g.nodes[c][key][:] for c in g.successors(n)]
        g.nodes[n][key] = _get_lca_characters(character_states, missing_states, unmodified_state)


def ancestral_characters(
    tdata: TreeData,
    characters_key: str = "characters",
    tree_key: str | None = None,
    missing_state: int | None = None,
    unmodified_state: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct ancestral character states using Camin-Sokal parsimony.

    Leaf character states are seeded from the tree's character matrix and the
    internal-node states are inferred bottom-up obeying Camin-Sokal parsimony
    (i.e. irreversibility). By default, states are written to the
    ``characters_key`` node attribute — the same name as the obsm character
    matrix — so leaves and internal nodes share a single, consistent attribute.
    Pass ``key_added`` to store the inferred states (and seeded leaf states)
    under a different node attribute.

    Args:
        tdata: The TreeData object to operate on.
        characters_key: The ``obsm`` key for the character matrix used to seed
            leaf states.
        tree_key: The ``obst`` key of the tree to use.
        missing_state: Missing-data value. Resolved from the tdata when ``None``.
        unmodified_state: Unmodified (uncut) state value. Resolved from the tdata
            when ``None``.
        key_added: Node attribute under which to store states. Defaults to
            ``characters_key`` when ``None``.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.
    """
    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    missing_states = _normalize_missing(_get_parameter(tdata, "missing_state", value=missing_state))
    unmodified = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    if isinstance(unmodified, (list, tuple, set)):
        unmodified = next(iter(unmodified))

    write_key = key_added if key_added is not None else characters_key
    character_matrix = _get_characters(tdata, characters_key)
    _seed_leaf_states(g, character_matrix, write_key)

    _reconstruct(g, missing_states, unmodified, write_key)

    _set_tree(tdata, g, tree_key)

    return tdata if copy else None
