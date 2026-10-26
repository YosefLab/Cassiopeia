"""Tests for the utilities in cassiopeia/mixins/utilities.py."""

import pandas as pd
import pytest

from cassiopeia.mixins import utilities


@pytest.mark.parametrize(("state", "expected"), [((1, 2), True), (1, False)])
def test_is_ambiguous_state(state, expected):
    assert utilities.is_ambiguous_state(state) is expected


@pytest.mark.parametrize(
    "state_array",
    [[0, (1, 2), 3, 4, 5], [0, 1, 2, 3, 4, 5]],
)
def test_unravel_ambiguous_states(state_array):
    assert utilities.unravel_ambiguous_states(state_array) == [0, 1, 2, 3, 4, 5]


def test_find_duplicate_groups():
    character_matrix = pd.DataFrame.from_dict(
        {
            "c1": [(5, 1), 0, 1, 2, 0],
            "c2": [(5, 1), 0, 1, 2, 0],
            "c3": [4, 0, 3, 2, -1],
            "c4": [-1, 4, 0, 2, 2],
            "c5": [0, 4, 1, 2, 2],
            "c6": [4, 0, 0, 2, (2, 1)],
            "c6_dup": [4, 0, 0, 2, (1, 2)],
        },
        orient="index",
        columns=["a", "b", "c", "d", "e"],
    )

    duplicated_mappings = utilities.find_duplicate_groups(character_matrix)

    for k, grp in [("c1", ("c1", "c2")), ("c6", ("c6", "c6_dup"))]:
        assert k in duplicated_mappings
        assert set(duplicated_mappings[k]) == set(grp)
