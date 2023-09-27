import functools
import importlib
from types import ModuleType
from typing import List, Optional, Tuple, Union


def is_ambiguous_state(state: Union[int, Tuple[int, ...]]) -> bool:
    """Determine whether the provided state is ambiguous.

    Note that this function operates on a single (indel) state.

    Args:
        state: Single, possibly ambiguous, character state

    Returns:
        True if the state is ambiguous, False otherwise.
    """
    return isinstance(state, tuple)


def try_import(module: str) -> Optional[ModuleType]:
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


def unravel_ambiguous_states(
    state_array: List[Union[int, Tuple[int, ...]]]
) -> List[int]:
    """Helper function to unravel ambiguous states.

    Args:
        A list of states, potentially containing ambiguous states.

    Returns:
        A list of unique states contained in the list.
    """
    all_states = [
        list(state) if is_ambiguous_state(state) else [state]
        for state in state_array
    ]
    return functools.reduce(lambda a, b: a + b, all_states)
