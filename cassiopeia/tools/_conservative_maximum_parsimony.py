from copy import deepcopy
from typing import Dict, Set

from cassiopeia.data import CassiopeiaTree


def maximum_parsimony(tree: CassiopeiaTree):
    tree = deepcopy(tree)
    tree.reconstruct_ancestral_characters()
    tree.set_character_states(tree.root, [0] * tree.n_character)
    return tree


def _compute_set_of_descendent_states(
    tree: CassiopeiaTree,
) -> Dict[str, Set[int]]:
    """
    Compact representation of the set of states descending from each node.

    Importantly:
    - Missing states are excluded.
    - A state of 0 indicates either a descending state of 0, or two positive
        descending states.
    """

    def normalize(set_of_states: Set[int]) -> Set[int]:
        """
        A state of 0 should indicate either a descending state of 0,
        or two positive descending states.
        """
        set_of_states -= {tree.missing_state_indicator}
        if 0 in set_of_states:
            return {0}
        elif len(set_of_states) >= 2:
            return {0}
        else:
            return set_of_states

    k = tree.character_matrix.shape[1]
    set_of_descendant_states_dict = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        node_states = tree.get_character_states(node)
        if tree.is_leaf(node):
            set_of_descendant_states = [
                normalize(set([node_states[i]])) for i in range(k)
            ]
        else:
            set_of_descendant_states = [
                normalize(
                    set(
                        sum(
                            [
                                list(set_of_descendant_states_dict[c][i])
                                for c in tree.children(node)
                            ],
                            [],
                        )
                    )
                )
                for i in range(k)
            ]
        set_of_descendant_states_dict[node] = set_of_descendant_states
    return set_of_descendant_states_dict


def conservative_maximum_parsimony(
    tree: CassiopeiaTree,
) -> CassiopeiaTree:
    """
    Conservative Maximum Parsimony Reconstruction.

    Only states that are unambiguous under MP are imputed.
    The tree is NOT modified in place: a new tree is returned.
    """
    tree = deepcopy(tree)
    tree.reconstruct_ancestral_characters()  # maximum parsimony algorithm
    tree.set_character_states(tree.root, [0] * tree.n_character)
    k = tree.character_matrix.shape[1]

    # First, the bottom-up traversal to determine the set of descendant states
    set_of_descendant_states_dict = _compute_set_of_descendent_states(tree)

    # Now, the top-down traversal to determine which nodes satisfy condition 5
    # of VMPR and condition 5 of VMPR.
    satisfies_vmpr_cond_5 = {node: [None] * k for node in tree.nodes}
    satisfies_cmpr_prop_5 = {node: [None] * k for node in tree.nodes}
    for node in tree.depth_first_traverse_nodes(postorder=False):
        if tree.is_leaf(node):
            continue
        for i in range(k):
            set_of_descendant_states = set_of_descendant_states_dict[node][i]
            # We first check if any of the first 4 conditions of the VMPR
            # theorem are satisfied
            if (
                tree.is_root(node)
                or 0 in set_of_descendant_states  # has at least two >= 0 states
                or len(set_of_descendant_states) == 0  # is {-1}
            ):
                satisfies_vmpr_cond_5[node][i] = False
                satisfies_cmpr_prop_5[node][i] = False
            else:
                # Has a unique positive state below, but we must check the
                # ancestor condition.
                assert len(set_of_descendant_states) == 1
                satisfies_vmpr_cond_5[node][i] = True
                s = list(set_of_descendant_states)[0]
                # Check if two children have s (g = v case)
                if (
                    sum(
                        [
                            s in set_of_descendant_states_dict[child][i]
                            for child in tree.children(node)
                        ]
                    )
                    >= 2
                ):
                    satisfies_cmpr_prop_5[node][i] = True
                # Check if the parent satisfies the property (g strict
                # ancestor of v case; because we are doing in-order
                # traversal, we can just check the parent of v)
                else:
                    parent = tree.parent(node)
                    satisfies_cmpr_prop_5[node][i] = satisfies_cmpr_prop_5[
                        parent
                    ][i]

    # Now set to -1 all the ambiguous MP states, which are those that satisfy
    # condition 5 of VMPR but not condition 5 of CMPR.
    cmpr_states_dict = {}
    for node in tree.depth_first_traverse_nodes(postorder=True):
        mp_states = tree.get_character_states(node)
        if tree.is_leaf(node):
            cmpr_states_dict[node] = mp_states
            continue
        cmpr_states = []
        for i in range(k):
            assert satisfies_vmpr_cond_5[node][i] is not None
            assert satisfies_cmpr_prop_5[node][i] is not None
            if (
                satisfies_vmpr_cond_5[node][i]
                and not satisfies_cmpr_prop_5[node][i]
            ):
                cmpr_states.append(
                    tree.missing_state_indicator
                )  # We wipe it out
            else:
                cmpr_states.append(mp_states[i])  # We keep it
        cmpr_states_dict[node] = cmpr_states
    tree.set_all_character_states(cmpr_states_dict)
    return tree
