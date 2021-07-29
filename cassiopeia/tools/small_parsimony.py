"""
This file contains several tools useful for using small-parsimony to analyze
phylogenies.

Amongst these tools are basic Fitch-Hartigan reconstruction, parsimony scoring,
and the FitchCount algorithm described in Quinn, Jones et al, Science (2021).
"""
from cassiopeia.mixins.errors import CassiopeiaTreeError
from typing import Optional

import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import CassiopeiaError


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
        root: Root from which to begin this refinement.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral states.
        label_key: Key to add that stores the maximum-parsimony assignment
            inferred from the Fitch-Hartigan top-down refinement.
        copy: Modify the tree in place or not.
    
    Returns:
        A new CassiopeiaTree if the copy is set to True, else None.
    """

    cassiopeia_tree = cassiopeia_tree.copy() if copy else cassiopeia_tree

    fitch_hartigan_bottom_up(cassiopeia_tree, meta_item, state_key)

    fitch_hartigan_top_down(
        cassiopeia_tree, root, state_key, label_key
    )

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
            of the meta data is not categorical.
    """

    if meta_item not in cassiopeia_tree.cell_meta.columns:
        raise CassiopeiaError("Meta item does not exist in the cassiopeia tree")

    meta = cassiopeia_tree.cell_meta[meta_item]

    if meta.dtype not in ['str', 'object']:
        raise CassiopeiaError("Meta item is not a categorical variable.")

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
        root: Root from which to begin this refinement.
        state_key: Attribute key that stores the Fitch-Hartigan ancestral states.
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
        root: Node to treat as the root.
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
    for (parent, child) in cassiopeia_tree.depth_first_traverse_edges(source=root):
        
        try:
            if cassiopeia_tree.get_attribute(parent, label_key) != cassiopeia_tree.get_attribute(child, label_key):
                parsimony += 1
        except CassiopeiaTreeError:
            raise CassiopeiaError(f"{label_key} does not exist for a node, "
                                "try running Fitch-Hartigan.")
    return parsimony
