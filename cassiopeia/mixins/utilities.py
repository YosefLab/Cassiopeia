import importlib
from types import ModuleType
from typing import Optional, Tuple, Union


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