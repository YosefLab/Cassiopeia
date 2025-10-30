"""Utilities for applying small-parsimony analyses to phylogenies."""

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from treedata import TreeData

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import (
    CassiopeiaError,
    CassiopeiaTreeError,
    FitchCountError,
)
from cassiopeia.typing import TreeLike
from cassiopeia.utils import (
    _get_digraph,
    depth_first_traverse_nodes_treelike,
    get_attribute_treelike,
    get_cell_meta,
    get_children_treelike,
    get_root,
    is_leaf_treelike,
    set_attribute_treelike,
)


def fitch_hartigan(
    tree: TreeLike,
    key: str,
    root: str | None = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
    treedata_key: str = None,
) -> TreeLike | None:
    """Run the Fitch-Hartigan algorithm.

    Performs the full Fitch-Hartigan small parsimony algorithm which, given
    a set of states for the leaves, infers the most-parsimonious set of states
    and returns a random solution that satisfies the maximum-parsimony
    criterion. The solution will be stored in the label key specified by the
    user (by default 'label'). This function will modify the tree in place
    if `copy=False`.

    Args:
        tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        key: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.
        treedata_key: If tree is a TreeData object, specify the key corresponding to the tree to process.

    Returns:
            A new CassiopeiaTree/TreeData/nx DiGraph + meta_df if the copy is set to True, else None.
    """
    tree = tree.copy() if copy else tree

    fitch_hartigan_bottom_up(tree, key, state_key)

    fitch_hartigan_top_down(tree, root, state_key, label_key)

    return tree if copy else None


def fitch_hartigan_bottom_up(
    tree: TreeLike,
    key: str,
    add_key: str = "S1",
    copy: bool = False,
    meta_df: pd.DataFrame | None = None,
    treedata_key: str = None,
) -> TreeLike | None:
    """Performs Fitch-Hartigan bottom-up ancestral reconstruction.

    Performs the bottom-up phase of the Fitch-Hartigan small parsimony
    algorithm. A new attribute called "S1" will be added to each node
    storing the optimal set of ancestral states inferred from this bottom-up
    algorithm. If copy is False, the tree will be modified in place.


    Args:
        tree: CassiopeiaTree object with cell meta data.
        key: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        meta_df: Optional DataFrame containing cell meta data. Only pass in if using networkx DiGraph.
        add_key: Key to add for bottom-up reconstruction
        copy: Modify the tree in place or not.
        treedata_key: If tree is a TreeData object, specify the key corresponding to the tree to process.


    Returns:
            A new CassiopeiaTree/TreeData/nx DiGraph + meta_df if the copy is set to True, else None.

    Raises:
            CassiopeiaError if the tree does not have the specified meta data
            or the meta data is not categorical.
    """
    if meta_df is None:
        meta_df = get_cell_meta(tree)

    if key not in meta_df.columns:
        raise CassiopeiaError("Key variable does not exist in the metadata for the tree object.")

    meta = meta_df[key]

    if is_numeric_dtype(meta):
        raise CassiopeiaError("Meta item is not a categorical variable.")

    if not is_categorical_dtype(meta):
        meta = meta.astype("category")

    tree = tree.copy() if copy else tree
    g, _ = _get_digraph(tree, treedata_key)
    g = g.copy() if copy else g

    for node in depth_first_traverse_nodes_treelike(g):
        if is_leaf_treelike(g, node):
            set_attribute_treelike(g, node, add_key, [meta.loc[node]])

        else:
            children = get_children_treelike(g, node)
            all_labels = np.concatenate(
                [get_attribute_treelike(g, child, add_key) for child in children]
            )

            states, frequencies = np.unique(all_labels, return_counts=True)

            S1 = states[np.where(frequencies == np.max(frequencies))]
            set_attribute_treelike(g, node, add_key, S1)

    if isinstance(tree, CassiopeiaTree) or isinstance(tree, TreeData):
        for node in g.nodes:
            set_attribute_treelike(tree, node, add_key, get_attribute_treelike(g, node, add_key))
        return tree if copy else None
    elif isinstance(tree, nx.DiGraph):
        return g if copy else None

    return tree if copy else None


def fitch_hartigan_top_down(
    tree: TreeLike,
    root: str | None = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
    treedata_key: str = None,
) -> TreeLike | None:
    """Run Fitch-Hartigan top-down refinement.

    Runs the Fitch-Hartigan top-down algorithm which selects an optimal solution
    from the tree rooted at the specified root.

    Args:
        tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.
        treedata_key: If tree is a TreeData object, specify the key corresponding to the tree to process.

    Returns:
            A new CassiopeiaTree/TreeData/nx DiGraph + meta_df if the copy is set to True, else None.

    Raises:
            A CassiopeiaTreeError if Fitch-Hartigan bottom-up has not been called
        or if the state_key does not exist for a node.
    """
    # assign root
    root = tree.root if (root is None) else root

    tree = tree.copy() if copy else tree

    g, _ = _get_digraph(tree, treedata_key)
    g = g.copy() if copy else g
    inferred_root = get_root(g)
    root = inferred_root if (root is None) else root

    for node in depth_first_traverse_nodes_treelike(g, source=root, postorder=False):
        if node == root:
            root_states = get_attribute_treelike(g, root, state_key)
            set_attribute_treelike(g, root, label_key, np.random.choice(root_states))
            continue

        parent = next(g.predecessors(node))
        parent_label = get_attribute_treelike(g, parent, label_key)
        optimal_node_states = get_attribute_treelike(g, node, state_key)

        if parent_label in optimal_node_states:
            set_attribute_treelike(g, node, label_key, parent_label)

        else:
            set_attribute_treelike(g, node, label_key, np.random.choice(optimal_node_states))

    if isinstance(tree, CassiopeiaTree) or isinstance(tree, TreeData):
        for n in g.nodes:
            set_attribute_treelike(tree, n, label_key, get_attribute_treelike(g, n, label_key))
        return tree if copy else None
    elif isinstance(tree, nx.DiGraph):
        return g if copy else None
    return tree if copy else None


def score_small_parsimony(
    tree: CassiopeiaTree | TreeData,
    key: str,
    root: str | None = None,
    infer_ancestral_states: bool = True,
    label_key: str | None = "label",
) -> int:
    """Computes the small-parsimony of the tree.

    Using the meta data stored in the specified cell meta column, compute the
    parsimony score of the tree.

    Args:
        tree: CassiopeiaTree object with cell meta data.
        key: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Node to treat as the root. Only the subtree below
            this node will be considered.
        infer_ancestral_states: Whether or not ancestral states must be inferred
            (this will be False if `fitch_hartigan` has already been called on
            the tree.)
        label_key: If ancestral states have already been inferred, this key
            indicates the name of the attribute they're stored in.

    Returns:
            The parsimony score.

    Raises:
            CassiopeiaError if label_key has not been populated.
    """
    tree = tree.copy()

    if infer_ancestral_states:
        fitch_hartigan(tree, key, root, label_key=label_key)

    parsimony = 0
    for parent, child in tree.depth_first_traverse_edges(source=root):
        try:
            if tree.get_attribute(parent, label_key) != tree.get_attribute(child, label_key):
                parsimony += 1
        except CassiopeiaTreeError as error:
            raise CassiopeiaError(
                f"{label_key} does not exist for a node, "
                "try running Fitch-Hartigan or passing "
                "infer_ancestral_states=True."
            ) from error
    return parsimony


def fitch_count(
    tree: CassiopeiaTree | TreeData,
    key: str,
    root: str | None = None,
    infer_ancestral_states: bool = True,
    state_key: str = "S1",
    unique_states: list[str] | None = None,
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
        tree: CassiopeiaTree object with a tree and cell meta data.
        key: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Node to treat as the root. Only the subtree below this node will
            be considered for the procedure.
        infer_ancestral_states: Whether or not to initialize the ancestral state
            sets with Fitch-Hartigan.
        state_key: If ancestral state sets have already been created, then this
            argument specifies what the attribute name is in the CassiopeiaTree
        unique_states: State space that can be optionally provided by the user.
            If this is not provided, we take the unique values in
            `cell_meta[key]` to be the state space.

    Returns:
            An MxM count matrix indicating the number of edges that contained a
            transition between two states across all equally parsimonious
            solutions returned by Fitch-Hartigan.
    """
    tree = tree.copy()

    if unique_states is None:
        unique_states = tree.cell_meta[key].unique()
    else:
        if len(np.setdiff1d(tree.cell_meta[key].unique(), unique_states)) > 0:
            raise FitchCountError(
                "Specified state space does not span the set of states that appear in the meta data."
            )

    if root != tree.root:
        tree.subset_clade(root)

    if infer_ancestral_states:
        fitch_hartigan_bottom_up(tree, key, add_key=state_key)

    # create mapping from nodes to integers
    bfs_postorder = [tree.root]
    for _, e1 in tree.breadth_first_traverse_edges():
        bfs_postorder.append(e1)

    node_to_i = dict(zip(bfs_postorder, range(len(bfs_postorder)), strict=False))
    label_to_j = dict(zip(unique_states, range(len(unique_states)), strict=False))

    N = _N_fitch_count(tree, unique_states, node_to_i, label_to_j, state_key)

    C = _C_fitch_count(tree, N, unique_states, node_to_i, label_to_j, state_key)

    M = pd.DataFrame(np.zeros((N.shape[1], N.shape[1])))
    M.columns = unique_states
    M.index = unique_states

    # create count matrix
    for s1 in unique_states:
        for s2 in unique_states:
            M.loc[s1, s2] = np.sum(
                C[
                    node_to_i[tree.root],
                    :,
                    label_to_j[s1],
                    label_to_j[s2],
                ]
            )

    return M


def _N_fitch_count(
    tree: CassiopeiaTree | TreeData,
    unique_states: list[str],
    node_to_i: dict[str, int],
    label_to_j: dict[str, int],
    state_key: str = "S1",
) -> np.ndarray:
    """Fill in the dynamic programming table N for FitchCount.

    Computes N[v, s], corresponding to the number of solutions below
    a node v in the tree given v takes on the state s.

    Args:
        tree: CassiopeiaTree object
        unique_states: The state space that a node can take on
        node_to_i: Helper array storing a mapping of each node to a unique
            integer
        label_to_j: Helper array storing a mapping of each unique state in the
            state space to a unique integer
        state_key: Attribute name in the CassiopeiaTree storing the possible
            states for each node, as inferred with the Fitch-Hartigan algorithm

    Returns:
            A 2-dimensional array storing N[v, s] - the number of
            equally-parsimonious solutions below node v, given v takes on
            state s
    """

    def _fill(v: str, s: str):
        """Helper function to fill in a single entry in N."""
        if tree.is_leaf(v):
            return 1

        children = tree.children(v)
        A = np.zeros(len(children))

        legal_states = []
        for i, u in zip(range(len(children)), children, strict=False):
            if s not in tree.get_attribute(u, state_key):
                legal_states = tree.get_attribute(u, state_key)
            else:
                legal_states = [s]

            A[i] = np.sum([N[node_to_i[u], label_to_j[sp]] for sp in legal_states])
        return np.prod([A[u] for u in range(len(A))])

    N = np.full((len(tree.nodes), len(unique_states)), 0.0)
    for n in tree.depth_first_traverse_nodes():
        for s in tree.get_attribute(n, state_key):
            N[node_to_i[n], label_to_j[s]] = _fill(n, s)

    return N


def _C_fitch_count(
    tree: CassiopeiaTree | TreeData,
    N: np.array,
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
        tree: CassiopeiaTree object
        N: N array computed during FitchCount storing the number of solutions
            below a node v given v takes on state s
        unique_states: The state space that a node can take on
        node_to_i: Helper array storing a mapping of each node to a unique
            integer
        label_to_j: Helper array storing a mapping of each unique state in the
            state space to a unique integer
        state_key: Attribute name in the CassiopeiaTree storing the possible
            states for each node, as inferred with the Fitch-Hartigan algorithm

    Returns:
            A 4-dimensional array storing C[v, s, s1, s2] - the number of
            transitions from state s1 to s2 below a node v given v takes on
            the state s.
    """

    def _fill(v: str, s: str, s1: str, s2: str) -> int:
        """Helper function to fill in a single entry in C."""
        if tree.is_leaf(v):
            return 0

        children = tree.children(v)
        A = np.zeros(len(children))
        LS = [[]] * len(children)

        for i, u in zip(range(len(children)), children, strict=False):
            if s in tree.get_attribute(u, state_key):
                LS[i] = [s]
            else:
                LS[i] = tree.get_attribute(u, state_key)

            A[i] = np.sum(
                [
                    C[
                        node_to_i[u],
                        label_to_j[sp],
                        label_to_j[s1],
                        label_to_j[s2],
                    ]
                    for sp in LS[i]
                ]
            )

            if s1 == s and s2 in LS[i]:
                A[i] += N[node_to_i[u], label_to_j[s2]]

        parts = []
        for i, u in zip(range(len(children)), children, strict=False):
            prod = 1

            for k, up in zip(range(len(children)), children, strict=False):
                fact = 0
                if up == u:
                    continue
                for sp in LS[k]:
                    fact += N[node_to_i[up], label_to_j[sp]]
                prod *= fact

            part = A[i] * prod
            parts.append(part)

        return np.sum(parts)

    C = np.zeros((len(tree.nodes), N.shape[1], N.shape[1], N.shape[1]))

    for n in tree.depth_first_traverse_nodes():
        for s in tree.get_attribute(n, state_key):
            for s1, s2 in itertools.product(unique_states, repeat=2):
                C[node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]] = _fill(n, s, s1, s2)

    return C
