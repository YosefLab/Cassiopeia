"""Utilities for applying small-parsimony analyses to phylogenies."""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cassiopeia.mixins.errors import (
    CassiopeiaError,
    FitchCountError,
)
from cassiopeia.typing import TreeLike
from cassiopeia.utils import (
    _get_cell_meta,
    _get_digraph,
    get_root,
)


def fitch_hartigan(
    tree: TreeLike,
    key: str,
    root: str | None = None,
    state_key: str = "S1",
    label_key: str = "label",
    meta_df: pd.DataFrame | None = None,
    tree_key: str = None,
    copy: bool = False,
) -> TreeLike | None:
    """Run the Fitch-Hartigan algorithm.

    Performs the full Fitch-Hartigan small parsimony algorithm which, given
    a set of states for the leaves, infers the most-parsimonious set of states
    and returns a random solution that satisfies the maximum-parsimony
    criterion. The solution will be stored in the label key specified by the
    user (by default 'label'). This function will modify the tree in place
    if `copy=False`.

    Args:
        tree: The tree to run the algorithm on.
        key: A column in the cell meta corresponding to a categorical variable.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key to store the Fitch-Hartigan ancestral state
            sets computed during the bottom-up pass.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        meta_df: Optional DataFrame containing cell meta data. Only pass in
            if using an nx.DiGraph.
        tree_key: If tree is a TreeData object, specify the key corresponding
            to the tree to process.
        copy: Modify the tree in place or not.

    Returns:
        A modified copy of the tree if copy=True, else None.

    Raises:
        CassiopeiaError if the tree does not have the specified meta data
        or the meta data is not categorical.
    """
    if meta_df is None:
        meta_df = _get_cell_meta(tree)

    if key not in meta_df.columns:
        raise CassiopeiaError("Key variable does not exist in the metadata for the tree object.")

    meta = meta_df[key]

    if is_numeric_dtype(meta):
        raise CassiopeiaError("Meta item is not a categorical variable.")

    if not isinstance(meta.dtype, pd.CategoricalDtype):
        meta = meta.astype("category")

    tree = tree.copy() if copy else tree
    g, tree_key = _get_digraph(tree, tree_key)

    _fitch_hartigan_bottom_up(g, meta, state_key)

    actual_root = root if root is not None else get_root(g)
    _fitch_hartigan_top_down(g, actual_root, state_key, label_key)

    return tree if copy else None


def _fitch_hartigan_bottom_up(g: nx.DiGraph, meta: pd.Series, add_key: str) -> None:
    """Bottom-up phase of Fitch-Hartigan on an nx.DiGraph.

    Args:
        g: Directed graph representing the tree.
        meta: Series mapping leaf node names to their observed states.
        add_key: Node attribute key to store the optimal state sets.
    """
    for node in nx.dfs_postorder_nodes(g):
        if g.out_degree(node) == 0:
            g.nodes[node][add_key] = [meta.loc[node]]
        else:
            children = list(g.successors(node))
            all_labels = np.concatenate([g.nodes[child][add_key] for child in children])
            states, frequencies = np.unique(all_labels, return_counts=True)
            g.nodes[node][add_key] = states[np.where(frequencies == np.max(frequencies))]


def _fitch_hartigan_top_down(
    g: nx.DiGraph,
    root: str,
    state_key: str,
    label_key: str,
) -> None:
    """Top-down refinement phase of Fitch-Hartigan on an nx.DiGraph.

    Args:
        g: Directed graph with state_key attribute set on all nodes.
        root: Root node to begin traversal from.
        state_key: Node attribute storing the optimal state sets.
        label_key: Node attribute to write the selected label to.
    """
    for node in nx.dfs_preorder_nodes(g, source=root):
        if node == root:
            g.nodes[node][label_key] = np.random.choice(g.nodes[node][state_key])
            continue

        parent = next(g.predecessors(node))
        parent_label = g.nodes[parent][label_key]
        optimal_states = g.nodes[node][state_key]

        if parent_label in optimal_states:
            g.nodes[node][label_key] = parent_label
        else:
            g.nodes[node][label_key] = np.random.choice(optimal_states)


def score_small_parsimony(
    tree: TreeLike,
    key: str,
    root: str | None = None,
    infer_ancestral_states: bool = True,
    label_key: str | None = "label",
    meta_df: pd.DataFrame | None = None,
    tree_key: str | None = None,
) -> int:
    """Computes the small-parsimony of the tree.

    Using the meta data stored in the specified cell meta column, compute the
    parsimony score of the tree.

    Args:
        tree: The tree to run the algorithm on.
        key: A column in the cell meta corresponding to a categorical variable.
        root: Node to treat as the root. Only the subtree below
            this node will be considered.
        infer_ancestral_states: Whether or not ancestral states must be inferred
            (this will be False if `fitch_hartigan` has already been called on
            the tree.)
        label_key: If ancestral states have already been inferred, this key
            indicates the name of the attribute they're stored in.
        meta_df: Optional DataFrame containing cell meta data. Only pass in
            if using an nx.DiGraph.
        tree_key: If tree is a TreeData object, specify the key corresponding
            to the tree to process.

    Returns:
        The parsimony score.

    Raises:
        CassiopeiaError if label_key has not been populated.
    """
    tree = tree.copy()

    if infer_ancestral_states:
        fitch_hartigan(tree, key, root, label_key=label_key, meta_df=meta_df, tree_key=tree_key)

    g, _ = _get_digraph(tree, tree_key)
    actual_root = root if root is not None else get_root(g)

    parsimony = 0
    for parent, child in nx.dfs_edges(g, source=actual_root):
        try:
            if g.nodes[parent][label_key] != g.nodes[child][label_key]:
                parsimony += 1
        except KeyError as error:
            raise CassiopeiaError(
                f"{label_key} does not exist for a node, "
                "try running Fitch-Hartigan or passing "
                "infer_ancestral_states=True."
            ) from error
    return parsimony


def fitch_count(
    tree: TreeLike,
    key: str,
    root: str | None = None,
    infer_ancestral_states: bool = True,
    state_key: str = "S1",
    unique_states: list[str] | None = None,
    tree_key: str | None = None,
) -> pd.DataFrame:
    """Runs the FitchCount algorithm.

    Performs the FitchCount algorithm for inferring the number of times that
    two states transition to one another across all equally-parsimonious
    solutions returned by the Fitch-Hartigan algorithm. The original algorithm
    was described in Quinn, Jones, et al, Science (2021). The output is an
    MxM count matrix, where the values indicate the number of times that
    m1 transitioned to m2 along an edge in a Fitch-Hartigan solution.
    To obtain probabilities P(m1 -> m2), divide each row by its row-sum.

    This procedure will only work on categorical data and will otherwise raise
    an error.

    Args:
        tree: The tree to run the algorithm on.
        key: A column in the cell meta corresponding to a categorical variable.
        root: Node to treat as the root. Only the subtree below this node will
            be considered for the procedure.
        infer_ancestral_states: Whether or not to initialize the ancestral state
            sets with Fitch-Hartigan.
        state_key: If ancestral state sets have already been created, then this
            argument specifies what the attribute name is in the tree.
        unique_states: State space that can be optionally provided by the user.
            If this is not provided, we take the unique values in
            `cell_meta[key]` to be the state space.
        tree_key: If tree is a TreeData object, specify the key corresponding
            to the tree to process.

    Returns:
        An MxM count matrix indicating the number of edges that contained a
        transition between two states across all equally parsimonious
        solutions returned by Fitch-Hartigan.
    """
    tree = tree.copy()
    meta_df = _get_cell_meta(tree)
    g, tree_key = _get_digraph(tree, tree_key)

    if unique_states is None:
        unique_states = meta_df[key].unique()
    elif len(np.setdiff1d(meta_df[key].unique(), unique_states)) > 0:
        raise FitchCountError(
            "Specified state space does not span the set of states that appear in the meta data."
        )

    if root is not None:
        g = g.subgraph(nx.descendants(g, root) | {root}).copy()
        actual_root = root
    else:
        actual_root = get_root(g)

    if infer_ancestral_states:
        _fitch_hartigan_bottom_up(g, meta_df[key], state_key)

    bfs_nodes = [actual_root] + [v for _, v in nx.bfs_edges(g, actual_root)]
    node_to_i = dict(zip(bfs_nodes, range(len(bfs_nodes)), strict=False))
    label_to_j = dict(zip(unique_states, range(len(unique_states)), strict=False))

    N = _N_fitch_count(g, unique_states, node_to_i, label_to_j, state_key)
    C = _C_fitch_count(g, N, unique_states, node_to_i, label_to_j, state_key)

    M = pd.DataFrame(
        np.zeros((len(unique_states), len(unique_states))),
        index=unique_states,
        columns=unique_states,
    )
    for s1 in unique_states:
        for s2 in unique_states:
            M.loc[s1, s2] = np.sum(C[node_to_i[actual_root], :, label_to_j[s1], label_to_j[s2]])

    return M


def _N_fitch_count(
    g: nx.DiGraph,
    unique_states: list[str],
    node_to_i: dict[str, int],
    label_to_j: dict[str, int],
    state_key: str = "S1",
) -> np.ndarray:
    """Fill in the dynamic programming table N for FitchCount.

    Computes N[v, s], corresponding to the number of solutions below
    a node v in the tree given v takes on the state s.

    Args:
        g: Directed graph with state_key attribute set on all nodes.
        unique_states: The state space that a node can take on.
        node_to_i: Mapping of each node to a unique integer.
        label_to_j: Mapping of each unique state to a unique integer.
        state_key: Node attribute storing the possible states for each node.

    Returns:
        A 2-dimensional array storing N[v, s].
    """

    def _fill(v: str, s: str) -> float:
        if g.out_degree(v) == 0:
            return 1
        children = list(g.successors(v))
        A = np.zeros(len(children))
        for i, u in enumerate(children):
            if s not in g.nodes[u][state_key]:
                legal_states = g.nodes[u][state_key]
            else:
                legal_states = [s]
            A[i] = np.sum([N[node_to_i[u], label_to_j[sp]] for sp in legal_states])
        return np.prod(A)

    N = np.full((len(g.nodes), len(unique_states)), 0.0)
    root = next(n for n in g.nodes if g.in_degree(n) == 0)
    for n in nx.dfs_postorder_nodes(g, source=root):
        for s in g.nodes[n][state_key]:
            N[node_to_i[n], label_to_j[s]] = _fill(n, s)

    return N


def _C_fitch_count(
    g: nx.DiGraph,
    N: np.ndarray,
    unique_states: list[str],
    node_to_i: dict[str, int],
    label_to_j: dict[str, int],
    state_key: str = "S1",
) -> np.ndarray:
    """Fill in the dynamic programming table C for FitchCount.

    Computes C[v, s, s1, s2], the number of transitions from state s1 to
    state s2 in the subtree rooted at v, given that state v takes on the
    state s.

    Args:
        g: Directed graph with state_key attribute set on all nodes.
        N: N array computed during FitchCount.
        unique_states: The state space that a node can take on.
        node_to_i: Mapping of each node to a unique integer.
        label_to_j: Mapping of each unique state to a unique integer.
        state_key: Node attribute storing the possible states for each node.

    Returns:
        A 4-dimensional array storing C[v, s, s1, s2].
    """

    def _fill(v: str, s: str, s1: str, s2: str) -> float:
        if g.out_degree(v) == 0:
            return 0

        children = list(g.successors(v))
        A = np.zeros(len(children))
        LS = [[]] * len(children)

        for i, u in enumerate(children):
            if s in g.nodes[u][state_key]:
                LS[i] = [s]
            else:
                LS[i] = g.nodes[u][state_key]

            A[i] = np.sum(
                [C[node_to_i[u], label_to_j[sp], label_to_j[s1], label_to_j[s2]] for sp in LS[i]]
            )

            if s1 == s and s2 in LS[i]:
                A[i] += N[node_to_i[u], label_to_j[s2]]

        parts = []
        for i, u in enumerate(children):
            prod = 1
            for k, up in enumerate(children):
                if up == u:
                    continue
                prod *= sum(N[node_to_i[up], label_to_j[sp]] for sp in LS[k])
            parts.append(A[i] * prod)

        return np.sum(parts)

    C = np.zeros((len(g.nodes), N.shape[1], N.shape[1], N.shape[1]))
    root = next(n for n in g.nodes if g.in_degree(n) == 0)
    for n in nx.dfs_postorder_nodes(g, source=root):
        for s in g.nodes[n][state_key]:
            for s1, s2 in itertools.product(unique_states, repeat=2):
                C[node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]] = _fill(n, s, s1, s2)

    return C
