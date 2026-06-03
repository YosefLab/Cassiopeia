from __future__ import annotations

import networkx as nx
import numpy as np

from cassiopeia.mixins import is_ambiguous_state
from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.typing import TreeLike
from cassiopeia.utils import (
    _get_character_matrix,
    _get_digraph,
)


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


def _resolve_missing_state_indicator(tree: TreeLike, missing_state_indicator: int | None) -> int:
    """Resolve the missing state indicator for *tree*.

    Returns *missing_state_indicator* when provided.  Otherwise reads it from the
    tree object (``CassiopeiaTree.missing_state_indicator`` or
    ``TreeData.uns['missing_state_indicator']``), defaulting to ``-1``.
    """
    if missing_state_indicator is not None:
        return missing_state_indicator

    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree

    if isinstance(tree, CassiopeiaTree):
        return tree.missing_state_indicator
    if isinstance(tree, TreeData):
        return tree.uns.get("missing_state_indicator", -1)
    return -1


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


def ancestral_characters(
    tree: TreeLike,
    characters_key: str = "characters",
    tree_key: str | None = None,
    missing_state_indicator: int | None = None,
    copy: bool = False,
) -> TreeLike | None:
    """Reconstruct ancestral character states using Camin-Sokal parsimony.

    Leaf character states are seeded from the tree's character matrix and the
    internal-node states are inferred bottom-up obeying Camin-Sokal parsimony
    (i.e. irreversibility).  States are written to the ``characters_key`` node
    attribute — the same name as the obsm character matrix — so leaves and
    internal nodes share a single, consistent attribute.  When ``tree`` is a
    plain :class:`~networkx.DiGraph`, leaves are assumed to already carry
    ``characters_key`` and are not re-seeded.

    Args:
        tree: The tree to operate on (CassiopeiaTree, TreeData, or nx.DiGraph).
        characters_key: Character matrix layer (CassiopeiaTree) or ``obsm`` key
            (TreeData) used to seed leaf states, and the node attribute under
            which inferred states are stored.
        tree_key: For TreeData, the ``obst`` key of the tree to use.
        missing_state_indicator: Missing-data value.  Resolved from the tree
            when ``None``.
        copy: If ``True``, operate on and return a copy of *tree*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tree* if ``copy=True``, else ``None``.
    """
    from treedata import TreeData

    tree = tree.copy() if copy else tree
    # TreeData stores frozen graphs; operate on a copy and write back.
    is_treedata = isinstance(tree, TreeData)
    g, tree_key = _get_digraph(tree, tree_key, copy=is_treedata)

    missing = _resolve_missing_state_indicator(tree, missing_state_indicator)
    if not isinstance(tree, nx.DiGraph):
        character_matrix = _get_character_matrix(tree, characters_key)
        _seed_leaf_states(g, character_matrix, characters_key)

    reconstruct_ancestral_characters(g, missing, characters_key)

    if is_treedata:
        tree.obst[tree_key] = g

    return tree if copy else None
