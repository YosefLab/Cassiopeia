import functools
import importlib
from types import ModuleType

import numpy as np


def is_ambiguous_state(state: int | tuple[int, ...]) -> bool:
    """Determine whether the provided state is ambiguous.

    Note that this function operates on a single (indel) state.

    Args:
        state: Single, possibly ambiguous, character state

    Returns
    -------
        True if the state is ambiguous, False otherwise.
    """
    return isinstance(state, tuple)


def try_import(module: str) -> ModuleType | None:
    """Helper function to import a possibly not-installed module.

    Args:
        module: Module to try and import

    Returns
    -------
        The imported module, if the module exists, or None
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None


def unravel_ambiguous_states(
    state_array: list[int | tuple[int, ...]]
) -> list[int]:
    """Helper function to unravel ambiguous states.

    Args:
        A list of states, potentially containing ambiguous states.

    Returns
    -------
        A list of unique states contained in the list.
    """
    all_states = [
        list(state) if is_ambiguous_state(state) else [state]
        for state in state_array
    ]
    return functools.reduce(lambda a, b: a + b, all_states)

def find_duplicate_groups(character_matrix) -> dict[str, tuple[str, ...]]:
    """Maps duplicated indices in character matrix to groups.

    Groups together samples in a character matrix if they have the same
    character states.

    Args:
        character_matrix: Character matrix, potentially with ambiguous states.

    Returns
    -------
        A mapping of a single sample name to the set of of samples that have
            the same character states.
    """
    character_matrix.index.name = "index"

     # convert to sets to support ambiguous states
    character_matrix_sets = character_matrix.copy()
    character_matrix_sets = character_matrix_sets.apply(
            lambda x: [
                set(s) if is_ambiguous_state(s) else {s}
                for s in x.values
            ],
            axis=0,
        ).apply(tuple, axis=1)
    is_duplicated = (
        character_matrix_sets.duplicated(keep=False)
    )
    unique_states = np.unique(character_matrix_sets[is_duplicated])
    duplicate_groups = [character_matrix_sets[character_matrix_sets == val].index.values for val in unique_states]
    duplicate_mappings =  {g[0]: tuple(g) for g in duplicate_groups}

    return duplicate_mappings
