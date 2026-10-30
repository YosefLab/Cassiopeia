import functools
import importlib
from types import ModuleType


def is_ambiguous_state(state: int | tuple[int, ...]) -> bool:
    """Determine whether the provided state is ambiguous.

    Note that this function operates on a single (indel) state.

    Args:
        state: Single, possibly ambiguous, character state

    Returns:
            True if the state is ambiguous, False otherwise.
    """
    return isinstance(state, tuple)


def try_import(module: str) -> ModuleType | None:
    """Helper function to import a possibly not-installed module.

    Args:
        module: Module to try and import

    Returns:
            The imported module, if the module exists, or None
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        return None


def unravel_ambiguous_states(state_array: list[int | tuple[int, ...]]) -> list[int]:
    """Helper function to unravel ambiguous states.

    Args:
        state_array: A list of states, potentially containing ambiguous states.

    Returns:
            A list of unique states contained in the list.
    """
    all_states = [list(state) if is_ambiguous_state(state) else [state] for state in state_array]
    return functools.reduce(lambda a, b: a + b, all_states)


def find_duplicate_groups(character_matrix) -> dict[str, tuple[str, ...]]:
    """Maps duplicated indices in character matrix to groups.

    Groups together samples in a character matrix if they have the same
    character states.

    Args:
        character_matrix: Character matrix, potentially with ambiguous states.

    Returns:
            A mapping of a single sample name to the set of of samples that have
            the same character states.
    """
    character_matrix.index.name = "index"

    # Build a hashable key per row so identical rows can be grouped in a single
    # linear pass. Ambiguous states (tuples) become frozensets so that order
    # within an ambiguous state does not matter and the key stays hashable.
    groups: dict[tuple, list] = {}
    for idx, row in zip(
        character_matrix.index.to_numpy(), character_matrix.to_numpy(), strict=False
    ):
        key = tuple(frozenset(s) if is_ambiguous_state(s) else s for s in row)
        groups.setdefault(key, []).append(idx)

    # Keep only groups with duplicates; the first occurrence is the representative
    # (matching ``drop_duplicates(keep="first")``).
    duplicate_mappings = {g[0]: tuple(g) for g in groups.values() if len(g) > 1}

    return duplicate_mappings
