"""
This file tests the utilities stored in cassiopeia/data/utilities.py
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
