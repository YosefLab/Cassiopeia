"""
This file contains several tools useful for using small-parsimony to analyze
phylogenies.

Amongst these tools are basic Fitch-Hartigan reconstruction, parsimony scoring,
and the FitchCount algorithm described in Quinn, Jones et al, Science (2021).
"""
from typing import Dict, List, Optional

import itertools
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype


from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import (
    CassiopeiaError,
    CassiopeiaTreeError,
    FitchCountError,
)


def fitch_hartigan(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    root: Optional[str] = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Run the Fitch-Hartigan algorithm.
    
    Performs the full Fitch-Hartigan small parsimony algorithm which, given
    a set of states for the leaves, infers the most-parsimonious set of states
    and returns a random solution that satisfies the maximum-parsimony
    criterion. The solution will be stored in the label key specified by the
    user (by default 'label'). This function will modify the tree in place
    if `copy=False`.

    Args:
        cassiopeia_tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.
    
    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.
    """

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    fitch_hartigan_bottom_up(cassiopeia_tree, meta_item, state_key)

    fitch_hartigan_top_down(cassiopeia_tree, root, state_key, label_key)

    return cassiopeia_tree if copy else None


def fitch_hartigan_bottom_up(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    add_key: str = "S1",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Performs Fitch-Hartigan bottom-up ancestral reconstruction.

    Performs the bottom-up phase of the Fitch-Hartigan small parsimony
    algorithm. A new attribute called "S1" will be added to each node
    storing the optimal set of ancestral states inferred from this bottom-up 
    algorithm. If copy is False, the tree will be modified in place.
     

    Args:
        cassiopeia_tree: CassiopeiaTree object with cell meta data.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        add_key: Key to add for bottom-up reconstruction
        copy: Modify the tree in place or not.

    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.

    Raises:
        CassiopeiaError if the tree does not have the specified meta data
            or the meta data is not categorical.
    """

    if meta_item not in cassiopeia_tree.cell_meta.columns:
        raise CassiopeiaError("Meta item does not exist in the cassiopeia tree")

    meta = cassiopeia_tree.cell_meta[meta_item]

    if is_numeric_dtype(meta):
        raise CassiopeiaError("Meta item is not a categorical variable.")

    if not is_categorical_dtype(meta):
        meta = meta.astype("category")

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    for node in cassiopeia_tree.depth_first_traverse_nodes():

        if cassiopeia_tree.is_leaf(node):
            cassiopeia_tree.set_attribute(node, add_key, [meta.loc[node]])

        else:
            children = cassiopeia_tree.children(node)
            if len(children) == 1:
                child_assignment = cassiopeia_tree.get_attribute(
                    children[0], add_key
                )
                cassiopeia_tree.set_attribute(node, add_key, [child_assignment])

            all_labels = np.concatenate(
                [
                    cassiopeia_tree.get_attribute(child, add_key)
                    for child in children
                ]
            )
            states, frequencies = np.unique(all_labels, return_counts=True)

            S1 = states[np.where(frequencies == np.max(frequencies))]
            cassiopeia_tree.set_attribute(node, add_key, S1)

    return cassiopeia_tree if copy else None


def fitch_hartigan_top_down(
    cassiopeia_tree: CassiopeiaTree,
    root: Optional[str] = None,
    state_key: str = "S1",
    label_key: str = "label",
    copy: bool = False,
) -> Optional[CassiopeiaTree]:
    """Run Fitch-Hartigan top-down refinement

    Runs the Fitch-Hartigan top-down algorithm which selects an optimal solution
    from the tree rooted at the specified root.

    Args:
        cassiopeia_tree: CassiopeiaTree that has been processed with the
            Fitch-Hartigan bottom-up algorithm.
        root: Root from which to begin this refinement. Only the subtree below
            this node will be considered.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral
            states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.

    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.

    Raises:
        A CassiopeiaTreeError if Fitch-Hartigan bottom-up has not been called
        or if the state_key does not exist for a node.
    """

    # assign root
    root = cassiopeia_tree.root if (root is None) else root

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    for node in cassiopeia_tree.depth_first_traverse_nodes(
        source=root, postorder=False
    ):

        if node == root:
            root_states = cassiopeia_tree.get_attribute(root, state_key)
            cassiopeia_tree.set_attribute(
                root, label_key, np.random.choice(root_states)
            )
            continue

        parent = cassiopeia_tree.parent(node)
        parent_label = cassiopeia_tree.get_attribute(parent, label_key)
        optimal_node_states = cassiopeia_tree.get_attribute(node, state_key)

        if parent_label in optimal_node_states:
            cassiopeia_tree.set_attribute(node, label_key, parent_label)

        else:
            cassiopeia_tree.set_attribute(
                node, label_key, np.random.choice(optimal_node_states)
            )

    return cassiopeia_tree if copy else None


def score_small_parsimony(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    root: Optional[str] = None,
    infer_ancestral_states: bool = True,
    label_key: Optional[str] = "label",
) -> int:
    """Computes the small-parsimony of the tree.

    Using the meta data stored in the specified cell meta column, compute the
    parsimony score of the tree.

    Args:
        cassiopeia_tree: CassiopeiaTree object with cell meta data.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
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

    cassiopeia_tree = cassiopeia_tree.copy()

    if infer_ancestral_states:
        fitch_hartigan(cassiopeia_tree, meta_item, root, label_key=label_key)

    parsimony = 0
    for (parent, child) in cassiopeia_tree.depth_first_traverse_edges(
        source=root
    ):

        try:
            if cassiopeia_tree.get_attribute(
                parent, label_key
            ) != cassiopeia_tree.get_attribute(child, label_key):
                parsimony += 1
        except CassiopeiaTreeError:
            raise CassiopeiaError(
                f"{label_key} does not exist for a node, "
                "try running Fitch-Hartigan or passing "
                "infer_ancestral_states=True."
            )
    return parsimony


def fitch_count(
    cassiopeia_tree: CassiopeiaTree,
    meta_item: str,
    root: Optional[str] = None,
    infer_ancestral_states: bool = True,
    state_key: str = "S1",
    unique_states: Optional[List[str]] = None,
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
        cassiopeia_tree: CassiopeiaTree object with a tree and cell meta data.
        meta_item: A column in the CassiopeiaTree cell meta corresponding to a
            categorical variable.
        root: Node to treat as the root. Only the subtree below this node will
            be considered for the procedure.
        infer_ancestral_states: Whether or not to initialize the ancestral state
            sets with Fitch-Hartigan.
        state_key: If ancestral state sets have already been created, then this
            argument specifies what the attribute name is in the CassiopeiaTree
        unique_states: State space that can be optionally provided by the user.
            If this is not provided, we take the unique values in
            `cell_meta[meta_item]` to be the state space.

    Returns:
        An MxM count matrix indicating the number of edges that contained a
            transition between two states across all equally parsimonious
            solutions returned by Fitch-Hartigan.
    """
    cassiopeia_tree = cassiopeia_tree.copy()

    if unique_states is None:
        unique_states = cassiopeia_tree.cell_meta[meta_item].unique()
    else:
        if (
            len(
                np.setdiff1d(
                    cassiopeia_tree.cell_meta[meta_item].unique(), unique_states
                )
            )
            > 0
        ):
            raise FitchCountError(
                "Specified state space does not span the set"
                " of states that appear in the meta data."
            )

    if root is not None and root != cassiopeia_tree.root:
        cassiopeia_tree.subset_clade(root)

    if infer_ancestral_states:
        fitch_hartigan_bottom_up(cassiopeia_tree, meta_item, add_key=state_key)

    # create mapping from nodes to integers
    bfs_postorder = [cassiopeia_tree.root]
    for (_, e1) in cassiopeia_tree.breadth_first_traverse_edges():
        bfs_postorder.append(e1)

    node_to_i = dict(zip(bfs_postorder, range(len(bfs_postorder))))
    label_to_j = dict(zip(unique_states, range(len(unique_states))))

    N = _N_fitch_count(
        cassiopeia_tree, unique_states, node_to_i, label_to_j, state_key
    )

    C = _C_fitch_count(
        cassiopeia_tree, N, unique_states, node_to_i, label_to_j, state_key
    )

    M = pd.DataFrame(np.zeros((N.shape[1], N.shape[1])))
    M.columns = unique_states
    M.index = unique_states

    # create count matrix
    for s1 in unique_states:
        for s2 in unique_states:
            M.loc[s1, s2] = np.sum(
                C[
                    node_to_i[cassiopeia_tree.root],
                    :,
                    label_to_j[s1],
                    label_to_j[s2],
                ]
            )

    return M


def _N_fitch_count(
    cassiopeia_tree: CassiopeiaTree,
    unique_states: List[str],
    node_to_i: Dict[str, int],
    label_to_j: Dict[str, int],
    state_key: str = "S1",
) -> np.array(int):
    """Fill in the dynamic programming table N for FitchCount.
    
    Computes N[v, s], corresponding to the number of solutions below
    a node v in the tree given v takes on the state s.

    Args:
        cassiopeia_tree: CassiopeiaTree object
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

        if cassiopeia_tree.is_leaf(v):
            return 1

        children = cassiopeia_tree.children(v)
        A = np.zeros((len(children)))

        legal_states = []
        for i, u in zip(range(len(children)), children):

            if s not in cassiopeia_tree.get_attribute(u, state_key):
                legal_states = cassiopeia_tree.get_attribute(u, state_key)
            else:
                legal_states = [s]

            A[i] = np.sum(
                [N[node_to_i[u], label_to_j[sp]] for sp in legal_states]
            )
        return np.prod([A[u] for u in range(len(A))])

    N = np.full((len(cassiopeia_tree.nodes), len(unique_states)), 0.0)
    for n in cassiopeia_tree.depth_first_traverse_nodes():
        for s in cassiopeia_tree.get_attribute(n, state_key):
            N[node_to_i[n], label_to_j[s]] = _fill(n, s)

    return N


def _C_fitch_count(
    cassiopeia_tree: CassiopeiaTree,
    N: np.array,
    unique_states: List[str],
    node_to_i: Dict[str, int],
    label_to_j: Dict[str, int],
    state_key: str = "S1",
) -> np.array(int):
    """Fill in the dynamic programming table C for FitchCount.
    
    Computes C[v, s, s1, s2], the number of transitions from state s1 to
    state s2 in the subtree rooted at v, given that state v takes on the
    state s. 

    Args:
        cassiopeia_tree: CassiopeiaTree object
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

        if cassiopeia_tree.is_leaf(v):
            return 0

        children = cassiopeia_tree.children(v)
        A = np.zeros((len(children)))
        LS = [[]] * len(children)

        for i, u in zip(range(len(children)), children):
            if s in cassiopeia_tree.get_attribute(u, state_key):
                LS[i] = [s]
            else:
                LS[i] = cassiopeia_tree.get_attribute(u, state_key)

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
        for i, u in zip(range(len(children)), children):
            prod = 1

            for k, up in zip(range(len(children)), children):
                fact = 0
                if up == u:
                    continue
                for sp in LS[k]:
                    fact += N[node_to_i[up], label_to_j[sp]]
                prod *= fact

            part = A[i] * prod
            parts.append(part)

        return np.sum(parts)

    C = np.zeros(
        (len(cassiopeia_tree.nodes), N.shape[1], N.shape[1], N.shape[1])
    )

    for n in cassiopeia_tree.depth_first_traverse_nodes():
        for s in cassiopeia_tree.get_attribute(n, state_key):
            for (s1, s2) in itertools.product(unique_states, repeat=2):
                C[
                    node_to_i[n], label_to_j[s], label_to_j[s1], label_to_j[s2]
                ] = _fill(n, s, s1, s2)

    return C
