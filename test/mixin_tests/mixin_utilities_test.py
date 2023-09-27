"""
This file tests the utilities stored in cassiopeia/data/utilities.py
"""

import unittest

import pandas as pd

from cassiopeia.mixins import utilities


class TestMixinUtilities(unittest.TestCase):
    def test_is_ambiguous_state(self):
        self.assertTrue(utilities.is_ambiguous_state((1, 2)))
        self.assertFalse(utilities.is_ambiguous_state(1))

    def test_unravel_states(self):
        state_array = [0, (1, 2), 3, 4, 5]
        self.assertListEqual(
            [0, 1, 2, 3, 4, 5], utilities.unravel_ambiguous_states(state_array)
        )
        
        state_array = [0, 1, 2, 3, 4, 5]
        self.assertListEqual(
            [0, 1, 2, 3, 4, 5], utilities.unravel_ambiguous_states(state_array)
        )

    def test_find_duplicated_character_states(self):

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
        
        expected_entries = [('c1', ('c1', 'c2')),
                            ('c6', ('c6', 'c6_dup'))]
        
        for k, grp in expected_entries:
            self.assertIn(k, list(duplicated_mappings.keys()))
            self.assertSetEqual(set(grp), set(duplicated_mappings[k]))


if __name__ == "__main__":
    unittest.main()
