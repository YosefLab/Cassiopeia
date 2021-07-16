from typing import Tuple, Union


def is_ambiguous_state(state: Union[int, Tuple[int, ...]]) -> bool:
    """Determine whether the provided state is ambiguous.

    Note that this function operates on a single (indel) state.

    Args:
        state: Single, possibly ambiguous, character state

    Returns:
        True if the state is ambiguous, False otherwise.
    """
    return isinstance(state, tuple)
